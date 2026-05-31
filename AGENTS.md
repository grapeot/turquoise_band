# AGENTS.md — turquoise_band 项目本地约定

## 工作语言
**整个工程的工作语言是中文。** 所有文档、报告、代码注释、commit message、曲线/图表的标题与输出说明，默认用中文。变量名、函数名、文件名等代码标识符仍用英文（遵循通用工程惯例）。

## 项目目标
重现月全食的绿松石带（turquoise band）。详见 `docs/PRD.md`、`docs/RFC.md`。终点是合成月食照片，但先用色相曲线（matplotlib）闭环验证物理对不对（红→青→白趋势），再上渲染。

## 环境
- `source .venv/bin/activate`（uv venv，Python 3.12）。装依赖用 `uv pip install`，不要用 `pip install`。
- 机器 M3 Ultra：32 CPU / 80 GPU / 512GB。渲染阶段用 MPS（torch），先 numpy 向量化。

## 验证哲学
每一步都要有能直接看的标量/曲线。先确认色相曲线形状对，再提分辨率、换真实数据、上渲染。每替换一个近似为真实数据，对比曲线确认没回归。
