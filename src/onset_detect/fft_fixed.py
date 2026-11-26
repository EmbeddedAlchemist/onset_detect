"""1024 点 Q15 实数 FFT（block-floating-point），与 target C 逐位等价。

这一步在做什么（给非信号处理背景的读者）
======================================
FFT 回答一个问题：**这一帧（64 ms）里，每个频率各有多少能量？**

    输入  1024 个时间样本（±32767 的整数）
    输出  513 个复数 X[k]，第 k 个对应频率 = k * 15.625 Hz
          k=0 → 0 Hz,  k=4 → 62.5 Hz,  k=51 → 797 Hz,  k=512 → 8000 Hz

处理步骤（`rfft_q15` 内部四步）
------------------------------
    第 1 步  加窗并打包：把 1024 个实数两两配成 512 个复数
             目的：实数输入有一半频点是重复的，只算一半 -> 省一半算力
    第 2 步  位反转 + 9 级蝶形（radix-2 DIT）
             目的：这就是"把时间变成频率"的核心运算
    第 3 步  实数 FFT 后处理
             目的：用 512 点复数的结果拼出 1024 点实数的 513 个频点
    第 4 步  求 |X|^2（不 sqrt）

为什么要 block-floating-point（BFP）
-----------------------------------
每级蝶形会让数值增长最多 1.414 倍，1024 点一路不缩放会增长约 1000 倍 -> int32 溢出。
CMSIS 的做法是"每级固定右移 1 位"，但 1024 点会因此损失 10 bit 有效位
（16 bit 音频只剩 6 bit，小信号被量化噪声淹没）。
本实现每级先看最大值，**超限才右移**，并记录一共移了多少位；
最后在对数域加回 `(shift-1)` 个倍频程即可精确还原。
（注意是"加"：右移把数值缩小了，要乘回去。）

内存（仅状态，不含 PCM 环）
    twiddle (m/2 complex)  : 256*2*4 B = 2 KB (Q15, 建议拷入 RAM)
    后处理 twiddle (m)     : 512*2*4 B = 4 KB -> 可只存 m/2 并用对称性, C 侧优化点
    位反转表 m             : 512*2 B   = 1 KB (可改为运行时计算, 省 RAM 费 CPU)
"""


from __future__ import annotations

import numpy as np

from .fxp import log2_sqrt_q8_8

# Q15 幅值 -> 对数域的标定。
#
# 定点路径得到的整数谱 X_int 与"真实"加窗 DFT X_true 的关系为:
#     X_int = X_true * 2^-(shift-1)
# 其中 shift = 1(打包右移) + 各级 BFP 移位 + 1(后处理右移)，
# 而 -1 是因为后处理的 >>1 被 X = E + W*O 的重建过程抵消。
# 因此对数域补偿是 **加** (shift-1) 个倍频程（移位缩小了数值，需要还原）。
def rfft_log_offset(shift: int) -> int:
    """返回该帧应加到 Q8.8 log 幅值上的偏移（单位 Q8.8）。"""
    return (int(shift) - 1) << 8


# 兼容旧名（= shift 为 1 时的偏移，等价于 0）
RFFT_LOG_BIAS_Q88 = 0


_BFP_LIMIT = 8192  # 每级蝶形前的幅度上限（留 1 bit 给 sqrt(2) 的旋转增长）


class RfftTables:
    """预计算表（Flash / RAM 常量）。"""

    __slots__ = ("n", "m", "log2m", "w_re", "w_im", "pw_re", "pw_im", "rev", "jrev")

    def __init__(self, n: int):
        if n < 8 or (n & (n - 1)) != 0:
            raise ValueError("n 必须是 2 的幂且 >= 8")
        self.n = n
        self.m = m = n // 2
        self.log2m = m.bit_length() - 1

        # 旋转因子 W_m^k = exp(-2*pi*i*k/m), k = 0..m/2-1
        k = np.arange(m // 2)
        ang = -2.0 * np.pi * k / m
        self.w_re = np.round(np.cos(ang) * 32767.0).astype(np.int32)
        self.w_im = np.round(np.sin(ang) * 32767.0).astype(np.int32)

        # 后处理旋转因子 W_n^k = exp(-2*pi*i*k/n), k = 0..m-1
        k2 = np.arange(m)
        ang2 = -2.0 * np.pi * k2 / n
        self.pw_re = np.round(np.cos(ang2) * 32767.0).astype(np.int32)
        self.pw_im = np.round(np.sin(ang2) * 32767.0).astype(np.int32)

        # 位反转索引
        bits = self.log2m
        idx = np.arange(m)
        rev = np.zeros(m, dtype=np.intp)
        for b in range(bits):
            rev = (rev << 1) | ((idx >> b) & 1)
        self.rev = rev
        # conj 对称: B[k] = conj(Z[(m-k) % m])
        self.jrev = (m - idx) % m


def hann_q15(n: int) -> np.ndarray:
    """周期 Hann 分析窗, Q15（峰值 32767）。C 侧可只存 (n/2+1) 点并镜像。

    作用: 让帧的两端平滑过渡到 0, 避免"硬截断"造成的频谱泄漏。
    副作用: 帧中央被强调 -> flux 峰值会比真实起振晚 16~24 ms（P1 标定后减掉）。
    """
    k = np.arange(n)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    return np.round(w * 32767.0).astype(np.int16)


def hann_float(n: int) -> np.ndarray:
    """浮点研究模式用的 Hann（仅离线对比，运行时禁止）。"""
    k = np.arange(n)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)).astype(np.float64)


def rfft_q15(xw: np.ndarray, t: RfftTables) -> tuple[np.ndarray, np.ndarray]:
    """实数 FFT（Q15 输入）。

    调用者必须先做两件事: 取 1024 点 + 乘 Hann 窗（见 hann_q15）。

    Parameters
    ----------
    xw : (..., n) 已加 Hann 窗的 Q15 样本（int16/int32）
    t  : 预计算表

    Returns
    -------
    s     : (..., m+1) int64, |X[k]|^2 (k = 0..n/2)
    shift : (...,)   int32, 该帧累计的右移位数

    调用方按 `log2_q8_8(s) >> 1  + rfft_log_offset(shift)` 得到 Q8.8 log 幅值。
    """
    x = np.asarray(xw, dtype=np.int32)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    b, n = x.shape
    m = t.m

    # --- 1. 打包成 m 点复序列 z[n] = x[2n] + j*x[2n+1] ---
    # 为什么: 实数序列的 FFT 有一半是冗余的(共轭对称), 把偶数/奇数样本分别塞进
    #        实部/虚部, 就能只做 m=n/2 点的复数 FFT, 最后再拼回来 -> 算力减半。
    # 为什么 >>1: 给后面的蝶形运算留 1 bit 余量, 否则第一级就可能溢出。
    re = (x[:, 0::2] >> 1).copy()
    im = (x[:, 1::2] >> 1).copy()
    shift = np.ones(b, dtype=np.int32)

    # --- 2. 位反转后做 m 点 radix-2 DIT（分治: 每级把问题折半）---
    # 一个蝶形: (a, b) -> (a + W*b, a - W*b)，W 是单位圆上的复数(旋转因子)
    # "位反转"是为了让就地(in-place)的迭代写法能对上分治后的数据顺序。
    re = re[:, t.rev]
    im = im[:, t.rev]

    half = 1
    while half < m:
        # 【关键】BFP: 只在数值快溢出时才整体右移 1 位, 并记录移了多少位。
        # 右移缩小了 2^shift 倍, 最后由 rfft_log_offset() 在对数域加回来。
        # 若像 CMSIS 那样每级无条件右移, 1024 点会白白损失 10 bit 有效位。
        mx = np.maximum(np.abs(re).max(axis=-1), np.abs(im).max(axis=-1))
        need = mx > _BFP_LIMIT
        if need.any():
            re = np.where(need[:, None], re >> 1, re)
            im = np.where(need[:, None], im >> 1, im)
            shift = shift + need.astype(np.int32)


        step = m // (2 * half)
        wr = t.w_re[::step][:half]
        wi = t.w_im[::step][:half]

        ar = re.reshape(b, -1, 2, half)
        ai = im.reshape(b, -1, 2, half)
        a0r, a1r = ar[:, :, 0, :], ar[:, :, 1, :]
        a0i, a1i = ai[:, :, 0, :], ai[:, :, 1, :]

        tr = (a1r * wr - a1i * wi) >> 15
        ti = (a1r * wi + a1i * wr) >> 15

        # 先算出四个临时量再写回（a0r 等是视图，不能边写边读）
        s0 = a0r + tr
        s1 = a0r - tr
        u0 = a0i + ti
        u1 = a0i - ti
        ar[:, :, 0, :] = s0
        ar[:, :, 1, :] = s1
        ai[:, :, 0, :] = u0
        ai[:, :, 1, :] = u1

        half <<= 1

    # --- 3. 实数 FFT 后处理 ---
    #   E 是"偶数样本序列"的谱, O 是"奇数样本序列"的谱,
    #   完整谱 X[k] = E[k] + W_n^k * O[k]   (k = 0..m-1), 再加 k=m 这一个点。
    #   A = Z[k], B = conj(Z[(m-k)%m])
    #   E = (A+B)/2,  O = -j(A-B)/2
    jr = t.jrev
    br = re[:, jr]
    bi = -im[:, jr]  # B.im

    er = (re + br) >> 1
    ei = (im + bi) >> 1
    dr = re - br
    di = im - bi  # = im + im[jr]

    orr = di >> 1  # O.re =  Di/2
    oi = -(dr >> 1)  # O.im = -Dr/2

    xr = er + ((orr * t.pw_re - oi * t.pw_im) >> 15)
    xi = ei + ((orr * t.pw_im + oi * t.pw_re) >> 15)

    # --- 4. 求 |X|^2（不做 sqrt, 后面用 log2(|X|^2)/2 代替 log2|X|）---
    s = xr.astype(np.int64) ** 2 + xi.astype(np.int64) ** 2
    xm = (re[:, 0] - im[:, 0]).astype(np.int64)
    s = np.concatenate([s, (xm * xm)[:, None]], axis=1)

    shift = shift + 1  # 后处理中的两次 >>1


    if single:
        return s[0], shift[0]
    return s, shift


def log_spectrum_q88(xw: np.ndarray, t: RfftTables) -> tuple[np.ndarray, int]:
    """调试用便捷函数（单帧）: 加窗样本 -> (Q8.8 log 幅值, shift)。"""
    s, sh = rfft_q15(xw, t)
    shift = int(np.asarray(sh).reshape(-1)[0])
    L = log2_sqrt_q8_8(s) + rfft_log_offset(shift)
    return L, shift


