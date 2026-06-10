# AGENTS.md — turquoise_band 项目本地约定

## 工作语言
**整个工程的工作语言是中文。** 所有文档、报告、代码注释、commit message、曲线/图表的标题与输出说明，默认用中文。变量名、函数名、文件名等代码标识符仍用英文（遵循通用工程惯例）。

## 项目目标
重现月全食的绿松石带（turquoise band）。详见 `docs/PRD.md`、`docs/RFC.md`。终点是合成月食照片，但先用色相曲线（matplotlib）闭环验证物理对不对（红→青→白趋势），再上渲染。

## 代码地图（动代码前先看）
每个 `src/` 模块 docstring 首行有【角色】标注，分四层：

- **权威物理管线**：`raytrace_eclipse.py`（正向 ray tracing 集成，`build_lut_from_raytrace` 是所有正式产出的物理出口）← `refraction_trace.py`（RK4 真折射）+ `curved_path.py`（弯曲路径消光）；文献对账/R-B 口径统一/模型卡走 `band_profile.py`。
- **共享基础**：`atmosphere` / `cross_sections` / `geometry` / `solar` / `color` / `data_loaders` / `radiative_transfer`（直线 τ，仅解析链用）。
- **解析对照/教学链**（物理已被权威管线取代，保留作对照与教学，勿用于新产出）：`pipeline`（L0 色相曲线）、`render`（L2 LUT 渲染 + 全项目共享显示链）、`render_rt`（点源 legacy 着色 + 权威 LUT 的查表着色器 `shade_disk_lut`）、`brute_ray_trace`（圆盘对照，曾称"金标准"）、`measure_ribbon`（legacy 口径，被 band_profile 取代）。
- **渲染应用**：`render_textured`（写实月盘，默认 raytrace 引擎）、`build_video`、`render_ablation`、`render_earth`、`export_video_data`、`starfield`（未接线素材）、`diag_*`（诊断脚本）。

## 入口与测试
- 写实月盘（SDR+HDR TIFF）：`python src/render_textured.py`（默认 d=40、权威 ray tracing LUT；`--engine pointsource` 为已废物理，仅历史对照）
- 权威管线自查：`python src/raytrace_eclipse.py`；光度曲线：`python scripts/render_photometric_profile.py`
- 视频：`python src/build_video.py`；ablation：`python src/render_ablation.py`
- HDR HEIC：`bash scripts/make_hdr.sh`（需环境变量 `HDR_REFERENCE_HEIC` 指向任一 iPhone 实拍 HEIC）
- 测试：`python -m pytest src/tests/ -v`（全套必须过；`-m "not slow"` 跳过慢测试，快速冒烟见 `test_render_smoke.py`）

## 文档路由
- `docs/working.md` — **当前权威记录**（Key Technical Decisions + Changelog + Lessons），改物理/口径前必读
- `docs/PHYSICS_AND_PITFALLS.md` — 物理方法与踩坑；`docs/MODEL_CARD.md` — 关键数字（由 `band_profile --model-card` 生成）
- `README.md` — 对外门面（状态/结论/复现命令）

## 环境
- `source .venv/bin/activate`（uv venv，Python 3.12）。装依赖用 `uv pip install`，不要用 `pip install`。
- 机器 M3 Ultra：32 CPU / 80 GPU / 512GB。全管线 numpy 矢量化 CPU（完整 ray tracing 秒级），未用 MPS。

## 验证哲学
每一步都要有能直接看的标量/曲线。先确认色相曲线形状对，再提分辨率、换真实数据、上渲染。每替换一个近似为真实数据，对比曲线确认没回归。
