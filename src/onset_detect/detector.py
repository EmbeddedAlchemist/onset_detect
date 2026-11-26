"""★唯一流式入口: process_block(int16[]) -> [Event]。

整条流水线在做什么（一句话版）
==============================
每 16 ms 取一段 64 ms 的音频 → 问"这段里 10 个频带各有多少能量" →
和上一段相减得到"能量涨了多少" → 涨得特别突出的地方就是一次起振。

逐级步骤（代码里对应 `_frame_flux` / `_on_frame` / `_eval_center`）
------------------------------------------------------------------
    [1] 加 Hann 窗        ：消除"硬截断"造成的频谱泄漏
    [2] 定点 FFT          ：得到 513 个频点的能量 |X|^2
    [3] 按 10 个子带求和   ：避免低频全带求和时贝斯稀释底鼓
    [4] 取 log2 (dB 域)    ：让"涨了多少"与绝对音量无关
    [5] 减去上一帧        ：得到谱通量 flux（只保留正值 = 只关心能量上升）
    [6] 直方图分位阈值     ：跟随段落响度自动升降
    [7] 局部极大 + 迟滞 + 最小间隔：把一条曲线变成一个干净的时间点

P0 版本到这里为止；P3 会在 [5] 之前插入"因果 HPSS 谐波抑制"，
P3/P4 会在 [7] 之后追加判别特征与打分（区分底鼓/贝斯/侧链恢复）。

约束
----
- 不前瞻: ±N 帧局部极大验证是**延迟**（N*hop），不是前瞻；事件时间戳回溯到极大帧。
- 无 float / 无 sqrt / 无除法（fixed=True 时）。
- 无动态分配: 所有状态在 __init__ 中一次性建立。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import BAND_NAMES, N_BANDS, band_bin_edges
from .fft_fixed import RfftTables, hann_float, hann_q15, rfft_q15

from .framer import Framer
from .fxp import log2_sqrt_q8_8

from .params import Params
from .quantile import HistQuantile


@dataclass(frozen=True)
class Event:
    """一次起振事件。

    字段含义:
        t             秒。等于"帧中心"时刻，**含一个常数滞后**（Hann 窗中央权重
                      最大导致 flux 峰晚于真实起振约 16~24 ms）。P1 会用标注数据
                      标定这个常数并减掉。
        sample        帧中心的绝对采样位置（最精确的整数表示）。
        frame         帧序号（第几帧，便于和 trace 对齐）。
        band          触发所在的子带（0=sub … 9=air）。同一个物理鼓点可能同时在
                      sub/kick 两个带触发，P3 的判别器会把它们合并成一个乐器事件。
        strength_q88  触发帧的 flux 超出阈值多少（Q8.8），可当作"把握程度"。
    """

    t: float           # 秒（帧中心，尚未做起始偏差标定）
    sample: int        # 帧中心的绝对采样位置
    frame: int         # 帧序号
    band: int          # 触发所在带
    strength_q88: int  # 触发帧 flux 超过阈值的量（Q8.8）

    @property
    def band_name(self) -> str:
        return BAND_NAMES[self.band]


class OnsetDetector:
    """流式鼓点检测器。单实例、非重入、无动态分配。

    状态说明（全部在 __init__ 一次性分配）:
        _framer       1280 点环形缓冲（把任意块长的输入切成固定 hop 的帧）
        tables        FFT 旋转因子/位反转表（只读，Flash 常量）
        _win_q15      Hann 窗（只读，Flash 常量）
        _prev_band_L  上一帧的 10 个带电平（用于做差得 flux）
        _hist         10 条 16 桶直方图（自适应阈值）
        _armed        10 个 bool：迟滞状态（是否允许再次触发）
        _last_hit     10 个帧号：上次触发时刻（最小间隔用）
        _ring         长度 2N+1 的延迟线（局部极大验证用）
    """

    def __init__(self, p: Params = Params()):
        if p.win % 2 or (p.win & (p.win - 1)):
            raise ValueError("win 必须为 2 的幂")
        if p.hop <= 0 or p.hop >= p.win:
            raise ValueError("0 < hop < win")
        self.p = p
        self.nb = N_BANDS

        self.tables = RfftTables(p.win)
        self._win_q15 = hann_q15(p.win).astype(np.int32)
        self._win_f = hann_float(p.win)

        edges = band_bin_edges(p.sr, p.win)
        self.band_edges = edges
        self._slices = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]

        self._hist = HistQuantile(
            self.nb,
            buckets=p.hist_buckets,
            shift=p.hist_shift,
            decay=p.hist_decay,
            q_num=p.thr_q_num,
            q_den=p.thr_q_den,
        )
        self._framer = Framer(p.win, p.hop)

        self._prev_band_L: np.ndarray | None = None
        self._armed = np.ones(self.nb, dtype=bool)
        self._last_hit = np.full(self.nb, -(1 << 30), dtype=np.int64)

        # 局部极大验证所需的短延迟线: (frame_idx, flux, thr, thr_rel)
        self._ring: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []

        self.n_frames = 0
        self.trace: dict[str, list] | None = None
        if p.trace:
            self.trace = {"band_L": [], "band_flux": [], "band_thr": []}

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def process_block(self, x: np.ndarray) -> list[Event]:
        """喂入任意长度 int16 块，返回本次新产生的事件（可能为空）。

        这是设备上唯一需要调用的函数。块长随便取（37/128/256/1000 都可以），
        输出事件序列**逐位相同**——这一点由 test_block_invariance 保证，
        也是"离线调参结果 == 设备行为"的前提。
        """
        out: list[Event] = []
        for frame in self._framer.push(x):
            out.extend(self._on_frame(frame))
        return out

    def flush(self) -> list[Event]:
        """流结束时把延迟线里剩下的事件判定完（离线分析用）。

        实时运行时一般不需要调用：它只是把"最后 N 帧还没确认完"的山顶补判，
        总共影响不超过 16~32 ms 的数据。
        """
        out: list[Event] = []
        la = self.p.peak_lookahead
        while len(self._ring) > la:
            out.extend(self._eval_center())
        self._ring.clear()
        return out

    def reset(self) -> None:
        self._framer.reset()
        self._hist.reset()
        self._prev_band_L = None
        self._armed[:] = True
        self._last_hit[:] = -(1 << 30)
        self._ring.clear()
        self.n_frames = 0

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _frame_flux(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一帧 -> (band_L, band_flux)，均为 Q8.8。

        处理步骤:
            [1] 加 Hann 窗: 避免硬截断造成频谱泄漏（否则底鼓会"漏"进贝斯带）
            [2] 定点 FFT  : 得到 513 个频点的 |X|^2
            [3] 按带求和   : 每个子带把 bin 的平方幅值加起来，再取 log2
                             -> 这就是该带当前的"电平"(dB)
            [4] 减上一帧   : flux = max(0, 本帧 - 上帧)
                             max(0, ...) 表示只关心"能量上升"，衰减不是起振
        """
        p = self.p
        if p.fixed:
            # [1] 加窗: x 是 Q15 整数, 窗也是 Q15, 乘完右移 15 位还原 Q15
            xw = (x.astype(np.int32) * self._win_q15) >> 15
            # [2] 定点实数 FFT -> 每个频点的 |X|^2 和本帧累计右移位数
            sq, shift = rfft_q15(xw, self.tables)
            # 对数域补偿: 右移把数值缩小了 2^shift, 这里加回 (shift-1) 个倍频程
            off = (int(shift) - 1) << 8
            band_L = np.empty(self.nb, dtype=np.int32)
            for b, sl in enumerate(self._slices):
                # [3] 带内求和 -> log2。注意: 先线性求和再取对数,
                #     而不是把每 bin 的对数相加(那是错的, logA+logB=logAB)
                band_L[b] = int(log2_sqrt_q8_8(int(sq[sl].sum()))) + off
        else:
            # 浮点研究模式: 与定点路径同尺度, 只用于对比验证
            spec = np.fft.rfft(x.astype(np.float64) * self._win_f)
            sq = spec.real * spec.real + spec.imag * spec.imag
            band_L = np.empty(self.nb, dtype=np.int32)
            for b, sl in enumerate(self._slices):
                tot = float(sq[sl].sum())
                band_L[b] = int(round(128.0 * np.log2(max(tot, 1.0))))

        # [4] 谱通量: 与上一帧相减, 负值归零
        if self._prev_band_L is None:
            flux = np.zeros(self.nb, dtype=np.int32)
        else:
            flux = np.maximum(0, band_L - self._prev_band_L)
        self._prev_band_L = band_L
        return band_L, flux.astype(np.int32)


    def _on_frame(self, x: np.ndarray) -> list[Event]:
        """每 16 ms 调用一次：算 flux → 更新阈值 → 送入延迟线等确认。"""
        p = self.p
        # [1]~[4] 加窗/FFT/分带/dB-flux
        band_L, flux = self._frame_flux(x)

        # [5] 把本帧 flux 计入直方图，再查询分位电平 -> 自适应阈值
        #     阈值 = 该带的"典型 flux 水平" + 4.1 dB
        #     这样安静段落阈值低、高潮段落阈值高，不需要手调魔法常数
        self._hist.update(flux)
        thr = self._hist.level() + p.thr_delta_q88
        # 迟滞的"释放阈值" = 触发阈值 * 3/5
        thr_rel = (thr * p.thr_hyst_num) // p.thr_hyst_den

        # 迟滞: 只有 flux 回落到释放阈值以下，该带才允许再次触发。
        # 没有这一步，一个"山包"上的小抖动会连续吐出好几个事件。
        self._armed |= flux < thr_rel

        # [6] 送进延迟线。要判断第 k 帧是不是"山顶"，必须看到第 k+1 帧，
        #     所以这里先存着；等延迟线满 2N+1 帧再回头判定中间那一帧。
        #     注意：这是"延迟"(用已到达的数据)，不是"前瞻"。
        self._ring.append((self.n_frames, flux, thr, thr_rel))
        self.n_frames += 1

        if self.trace is not None:
            self.trace["band_L"].append(band_L)
            self.trace["band_flux"].append(flux)
            self.trace["band_thr"].append(thr)

        out: list[Event] = []
        if len(self._ring) > 2 * p.peak_lookahead:
            out.extend(self._eval_center())
        return out


    def _eval_center(self) -> list[Event]:
        """判定延迟线正中间那一帧：它是不是一个"山顶"，且满足所有物理约束。"""
        p = self.p
        ci = p.peak_lookahead
        idx, flux, thr, thr_rel = self._ring[ci]

        # [7a] 超阈值 + 该带已"武装"(迟滞允许)
        cond = (flux > thr) & self._armed

        # [7b] 局部极大: 比左边所有邻居都高或相等, 比右边所有邻居都严格高。
        #      两边不对称是为了避免"平台"(连续几帧数值相同)被重复触发。
        for j in range(len(self._ring)):
            if j == ci:
                continue
            if j < ci:
                cond &= flux >= self._ring[j][1]
            else:
                cond &= flux > self._ring[j][1]

        # [7c] 最小间隔: 同一带 45 ms 内不可能真的打两次（物理下限）。
        #      不能设太长: 128 BPM 的 16 分音符间隔只有 117 ms。
        cond &= (idx - self._last_hit) >= p.min_ioi_frames

        out: list[Event] = []
        if cond.any():
            if not p.events_per_band:
                bands = [int(np.argmax(np.where(cond, flux, -1)))]
            else:
                bands = np.flatnonzero(cond).tolist()
            # 事件时间戳 = 该帧的**中心**采样位置。
            # 注意它还含一个常数滞后（Hann 窗导致 flux 峰偏晚 16~24 ms），
            # P1 会用标注数据标定后统一减掉。
            sample = idx * p.hop + p.win // 2
            for b in bands:
                self._last_hit[b] = idx
                self._armed[b] = False
                out.append(
                    Event(
                        t=sample / p.sr,
                        sample=int(sample),
                        frame=int(idx),
                        band=int(b),
                        strength_q88=int(flux[b] - thr[b]),
                    )
                )

        self._ring.pop(0)
        return out


