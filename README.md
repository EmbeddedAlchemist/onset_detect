# onset_detect

面向嵌入式（16 kHz / 16 bit，Cortex-M3 ~ RV32IM 级，无 FPU）的**流式**鼓点检测与节拍预测前端。

核心原则：**原型即实现**。Python 代码按定点 C 的语义书写（Q15 / Q8.8、无 float、无 sqrt、无除法、
无动态分配、单实例非重入），并有逐级对拍测试，保证移植时不会"算法变了"。

- 设计文档：[`docs/architecture.md`](docs/architecture.md)
- **零基础信号处理解说**：[`docs/dsp_explained.md`](docs/dsp_explained.md)（每一步在做什么、为什么、不做会怎样）
- 参数集中管理：`src/onset_detect/params.py`（每项都标注了物理作用）


## 代码 ↔ 文档对照

| 想了解 | 看这个文件 | 对应解说章节 |
|---|---|---|
| 声音怎么变成数字、环形缓冲为什么是 1280 | `framer.py` | §0、§1 |
| 为什么加窗、FFT 在算什么 | `fft_fixed.py` | §2、§3 |
| 为什么要取对数（dB）、怎么用整数算 log2 | `fxp.py` | §4 |
| 为什么分 10 个带、每个带是什么乐器 | `bands.py` | §5 |
| 谱通量 flux 的定义与局限 | `detector._frame_flux()` | §6 |
| 自适应阈值/分位数 | `quantile.py` | §7 |
| 局部极大、迟滞、最小间隔 | `detector._eval_center()` | §8 |
| 时间戳与延迟预算 | `detector.py` 模块头 | §9 |
| Q15 定点、block floating point | `fxp.py` / `fft_fixed.py` | §10 |
| 底鼓/贝斯/侧链在图上长什么样 | — | §11 |

## 编码约定

1. **每个处理步骤都要写明"在做什么 + 为什么 + 不做会怎样"**（中文注释）。
2. 新增模块必须先确定它在 §1–§11 中的位置，再写代码。
3. 运行时路径禁止 float / sqrt / 除法 / 动态分配；调试与绘图可以用。
4. 任何算法改动都必须先过 `pytest`（含块长不变性、定点/浮点一致性）。

## 怎么看图

两个可视化工具（`dump_analysis.py` / `show_envelopes.py`）的顶部面板现在是**波形 + 打击泳道**：

```
y=1.20  ┌──────────────────────────────────────────┐
        │▏  ▏▏   ▏ ▏    ▏▏  ▏   ▏▏    ▏  ▏▏   ▏   │ ← 打击泳道（每个 hit 一根短竖线）
y=1.05  ├──────────────────────────────────────────┤
        │                                          │
   0.0  │        ～～～ 波形 ～～～                  │
        │                                          │
y=-1.0  └──────────────────────────────────────────┘
```

- **颜色**：红=只触发低频带（底鼓候选）、青=只触发高频带（hi-hat/镲）、橙=低频+高频同时（底鼓带 click / 军鼓）、黄=其它（贝斯/混合）
- 鼠标移上去会显示"最近的打击 + 10 位指纹 + 猜测标签"
- 左键单击会在终端打印该时刻的打击详情（哪些带、各多少 dB）
- 想看每个带各自的原始事件（很密）：`--per-band`



```bash
python -m pip install -e ".[dev]"

# 交互式查看（弹出窗口：可缩放/平移，鼠标移动显示时间与最强带，单击打印附近事件）
python tools/dump_analysis.py "test_case_日不落-蔡依林.wav"

# 直接定位到某一段（之后仍可在窗口里继续缩放）
python tools/dump_analysis.py "test_case_日不落-蔡依林.wav" --zoom 12 24

# 需要图片时：加 --out 会同时导出 PNG；加 --no-show 则只导出不弹窗
python tools/dump_analysis.py "test_case_日不落-蔡依林.wav" --out artifacts/dump_full.png --no-show

# 看"频带指纹"：把每带事件合并成打击，统计有哪几种典型模式
python tools/band_patterns.py "test_case_日不落-蔡依林.wav"

# 看"乐器强度包络"：3 条 onset strength 曲线 + 自相关 BPM 自检（下游节拍跟踪的输入）
python tools/show_envelopes.py "test_case_日不落-蔡依林.wav"

# 测试
python -m pytest -q
```

### 窗口打不开？

如果运行后提示 `没有可用的 GUI 后端` 或只生成了 `artifacts/dump.png`，
说明当前 Python 没有图形后端（本项目的 venv 就没有编译 tkinter）。
装一个纯 pip 的 Qt 后端即可：

```bash
python -m pip install PyQt5
```

## 当前状态（P0 完成）

| 模块 | 内容 | 状态 |
|---|---|---|
| `fxp.py` | Q15/Q8.8 原语、CLZ 型 log2、免 sqrt 幅值 | ✅ |
| `fft_fixed.py` | 1024 点 Q15 实数 FFT（block-floating-point） | ✅ |
| `framer.py` | 1280 点环 + 跳帧（零拷贝、块长不变） | ✅ |
| `bands.py` | 10 个对数子带（bin 常量表） | ✅ |
| `quantile.py` | 16 桶整数直方图分位 | ✅ |
| `detector.py` | `process_block(int16[]) -> [Event]` | ✅（基线） |
| `hits.py` | 事件 → 打击（`Hit` 频带指纹）+ 流式 `HitMerger` | ✅ |
| `envelope.py` | 10 带 flux → pulse/snare/hat 三条强度包络 + BPM 自检 | ✅ |
| `tools/dump_analysis.py` | 波形+**打击泳道** / 带电平 / 带 flux / 原始事件(`--per-band`) | ✅ |
| `tools/band_patterns.py` | 指纹分布 + 带间共现 + 孤立事件诊断 | ✅ |
| `tools/show_envelopes.py` | 三条包络 + 打击分类 + 自相关 BPM | ✅ |
| 测试 | log2 精度、FFT 对拍、块长不变性、定点/浮点一致、hit 合并 | ✅ 27 passed |

实测：228.5 s 全曲 **9.5 s** 完成（**24× 实时**，纯 Python + 定点整数运算路径）。

## 路线图

- **P1** 评估闭环：标注工具 + `eval_onsets.py`（分乐器 P/R/F1、bass→kick 误报率、时间偏差、延迟）
- **P2** 自适应阈值升级（高/低分位 + 相对峰值）+ 时间戳偏差标定
- **P3** 因果 HPSS + 判别特征（A1 重心比 / B 衰减 / C click 同步）+ Q4.12 线性打分
- **P4** 侧链专项（软门控 + 部分集 L1）
- **P5** 多风格泛化
- **P6** 定点黄金向量 + C 端逐级对拍 + 资源实测
