"""整数定点原语（Q15 / Q8.8 / Q4.12），与 target C 实现语义一一对应。

本文件为什么长这样（给非信号处理背景的读者）
============================================
目标 MCU 没有硬件浮点单元（FPU）。一个 float 乘法要靠软件模拟，约 50~100 条指令，
而整数乘法只要 1~2 条。整条处理链每秒要做几百万次乘加，所以必须全程用整数。

三个"必须避开"的运算，以及替代方案:

    想要            为什么不能用            本文件的做法
    ------------------------------------------------------------------
    float 乘加      无 FPU, 慢 50~100 倍     Q15 定点乘法 mul_q15()
    sqrt(幅值)      同上 + 耗时             直接算 |X|^2, 最后 log/2 (数学等价)
    除法            同上 + 周期长            全部改到对数域做减法
    排序求中位数     插入排序太贵             直方图分位数 (见 quantile.py)

定点约定:
    Q15     : 1.0 == 1 << 15        (音频样本、窗、旋转因子)
    Q8.8    : 对数域, 1.0 倍频程 == 1 << 8   (1 LSB = 1/256 oct = 0.0234 dB)
    能量/幅值: 一律以整数平方幅值 |X|^2 参与运算 (避免 sqrt),
              再用 log2 转成对数域 (让"变化量"与绝对音量无关)
"""


from __future__ import annotations

import numpy as np

INT16_MIN = -(1 << 15)
INT16_MAX = (1 << 15) - 1

# log2(1 + k/16) * 256, k = 0..16  (Q8.8, 供 log2_q8_8 的 4-bit 插值使用)
LOG2_LUT = np.array(
    (0, 22, 44, 63, 82, 100, 118, 134, 150, 165, 179, 193, 207, 220, 232, 244, 256),
    dtype=np.int32,
)


# --------------------------------------------------------------------------- #
# 基本运算
# --------------------------------------------------------------------------- #
def sat16(x: np.ndarray) -> np.ndarray:
    """饱和到 int16（C 侧对应 __SSAT(x, 16)）。"""
    return np.clip(x, INT16_MIN, INT16_MAX).astype(np.int16)


def mul_q15(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Q15 * Q15 -> Q15（截断，与 C 的 `(a*b) >> 15` 完全一致）。

    要求 |a| <= 32767 且 |b| <= 32767；中间量 |a*b| <= 2^30 落在 int32。
    """
    return (np.asarray(a, dtype=np.int32) * np.asarray(b, dtype=np.int32)) >> 15


def mul_q15_add(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """a*b + c，先 shift 再相加（C 侧对应 SMLAWx / 手写 mul+add）。"""
    return mul_q15(a, b) + np.asarray(c, dtype=np.int32)


def q88_to_oct(v):
    """Q8.8 -> 倍频程（浮点，仅供调试/绘图）。"""
    return np.asarray(v, dtype=np.float64) / 256.0


def q88_to_db(v):
    """Q8.8（log2 尺度）-> dB（仅供调试/绘图）。"""
    return np.asarray(v, dtype=np.float64) * (20.0 * np.log10(2.0) / 256.0)


def db_to_q88(db):
    """dB -> Q8.8（仅供参数换算，运行时不要调用）。"""
    return int(round(np.asarray(db, dtype=np.float64) * 256.0 / (20.0 * np.log10(2.0))))


# --------------------------------------------------------------------------- #
# log2（CLZ + 16 段 LUT + 4 bit 线性插值）
# --------------------------------------------------------------------------- #
def _exp2(x: np.ndarray) -> np.ndarray:
    """floor(log2(x)), x >= 1。等价于 C 的 `31 - __builtin_clz(x)`。

    np.frexp 对 <= 2^53 的整数是精确的, 因此该实现与 CLZ 完全一致。
    """
    m, e = np.frexp(np.asarray(x, dtype=np.float64))
    return (e - 1).astype(np.int64)


def log2_q8_8(x: np.ndarray) -> np.ndarray:
    """log2(x) 以 Q8.8 返回（x 为整数 >= 1）。

    处理步骤（对应 target C 的四行）
    --------------------------------
    第 1 步  求指数 e = floor(log2(x))
             做法: 数二进制最高位的位置, C 里就是 `31 - __builtin_clz(x)`（一条指令）。
             为什么: 任何整数都能写成 x = 2^e * (1 + t), 0 <= t < 1。

    第 2 步  取出尾数 t 的 12 位定点表示
             做法: 把 x 左移 12 位再右移 e 位, 减掉 4096, 得到 t12 ∈ [0, 4096)。
             为什么: 后面的查找表只需要尾数, 12 位足够(误差 0.00035 倍频程)。

    第 3 步  用 16 段查找表 + 线性插值算 log2(1 + t)
             做法: 高 8 位选段(17 项表), 低 8 位做插值, 插值带四舍五入。
             为什么: 直接算 log 要么用 float, 要么用几十项泰勒展开; 查表只要 ~10 条指令。

    第 4 步  整数部分 + 小数部分拼起来: (e << 8) + v
             结果单位是 Q8.8, 即 1.0 倍频程 = 256, 1 LSB = 0.0234 dB。

    实测精度: 最大误差 0.0044 倍频程 = 0.038 dB（远小于 3~6 dB 的检测阈值）。
    """

    x = np.maximum(np.asarray(x), 1).astype(np.int64)
    e = _exp2(x)
    t12 = ((x << 12) >> e) - 4096  # 尾数 [0, 4096)
    k = (t12 >> 8).astype(np.intp)
    rem = (t12 & 255).astype(np.intp)
    # 插值带四舍五入, 去掉 (-0.5 LSB) 的系统性偏置
    v = LOG2_LUT[k] + ((((LOG2_LUT[k + 1] - LOG2_LUT[k]) * rem) + 128) >> 8)
    return ((e << 8) + v).astype(np.int32)




# --------------------------------------------------------------------------- #
# 幅值 / 能量（免 sqrt）
#
# 为什么要"平方幅值"而不是"幅值":
#   我们真正想要的是每个频点的强度 |X| = sqrt(re^2 + im^2)。
#   但 sqrt 在 MCU 上很贵(20~50 周期), 而下一步反正要取对数, 于是:
#       log2|X| = 0.5 * log2(|X|^2)
#   用 |X|^2 做对数再除以 2, 数学上完全等价。
#   C 侧注意: re^2 + im^2 必须用 32x32->64 乘法(mulh / SMULL), 否则会溢出。
# --------------------------------------------------------------------------- #
def log2_sqrt_q8_8(s: np.ndarray) -> np.ndarray:
    """256 * log2(sqrt(s))，四舍五入。

    作用: 把"平方幅值/能量"变成"对数幅值"(Q8.8)。
    相当于一次免费的开平方: 在指数域除以 2 就等于在数值域开平方。
    C 侧: return (log2_q8_8(s) + 1) >> 1;
    """
    return (log2_q8_8(s) + 1) >> 1


def sq_mag(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """|X|^2，int64（C 侧: 32x32->64 `mulh/mul` 或 `SMULL`）。

    FFT 每个频点是复数 (re, im)，其能量就是 re^2 + im^2。
    必须用 64 位装: 单边最大 ~40000 时平方已有 1.6e9，两个相加会溢出 int32。
    """
    a = np.asarray(re, dtype=np.int64)
    b = np.asarray(im, dtype=np.int64)
    return a * a + b * b


def log_mag_q8_8(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """256 * log2(|X|)，用平方幅值实现，完全避免 sqrt。

    仅用于调试/可视化。检测主路径不需要逐 bin 求对数——
    只要每个子带先求和、再求一次对数即可（省下 511 次 log）。
    """
    return log2_sqrt_q8_8(sq_mag(re, im))





# --------------------------------------------------------------------------- #
# 包络跟踪（快攻慢释，纯移位，无乘法/除法）
# --------------------------------------------------------------------------- #
class AsymFollower:
    """一阶非对称包络跟踪（快攻慢释）。P3 的时域 click 检测会用。

    作用: 跟踪一个信号的"最近水平"，上升快、下降慢。
    为什么要非对称: 起振要立刻抓到（attack 快），
                   而电平下降要慢慢跟（release 慢），否则会把同一个鼓点的衰减
                   误判成"电平很低"从而抬高灵敏度、造成重复触发。

    实现只用移位, 不用乘除法:
        d = target - state;
        state += (d > 0) ? (d >> atk) : (d >> rel);

    atk 越小跟得越快（atk=2 表示时间常数约 4 帧）。
    """

    __slots__ = ("state", "atk", "rel")

    def __init__(self, atk: int = 2, rel: int = 6, init: int = 0):
        self.state = np.int32(init)
        self.atk = atk
        self.rel = rel

    def step(self, target):
        d = np.asarray(target, dtype=np.int32) - self.state
        self.state = self.state + np.where(d > 0, d >> self.atk, d >> self.rel)
        return self.state
