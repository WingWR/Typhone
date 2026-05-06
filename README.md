# AI4S Typhoon PINN Visualization

## 项目介绍

本项目是一个基于 React + Flask 的台风路径预测与气象场可视化系统，面向 AI for Science 中的气候建模方向。项目当前已经接入 PINN-v1 方案：后端接收台风历史观测数据，通过 Physics-Informed Neural Network 预测未来台风位置、风速和气压，并在前端展示 PINN 预测轨迹、baseline 轨迹、真实轨迹和误差指标。

项目核心思想是把数据驱动预测和物理约束结合起来。PINN 训练时不仅拟合真实轨迹数据，还加入速度一致性、惯性约束、科里奥利力约束、风速-气压关系和近岸衰减约束，使模型预测更符合台风运动的物理规律。

当前项目已经完成：

- 独立训练模块：`train/train_pinn.py`
- PINN 模型定义：`backend/models/pinn_model.py`
- 模型权重目录：`backend/models/weights/`
- 后端推理层：`backend/logic/pinn_inference.py`
- 物理工具模块：`backend/utils/physics_engine.py`
- 预测接口接入：`POST /api/predict_typhoon`
- 前端多轨迹展示：PINN、baseline、actual
- 前端指标展示：data loss、physics loss、轨迹误差、风速误差、气压误差

如果没有训练好的模型权重，系统不会崩溃，会自动回退到线性平滑 baseline 预测。此时接口仍然可用，但 `model_name` 会显示为 `rule_baseline`，`inference_mode` 会显示为 `linear_fallback`。

## 运行与训练

安装后端依赖：

```powershell
python -m pip install -r backend\requirements.txt
```

推荐训练命令：

```powershell
python train\train_pinn.py --dataset train\CH2025BST_pinn_dataset.csv --epochs 200 --batch-size 32 --sequence-length 4 --hidden-dim 128 --lr 1e-3
```

训练脚本会按 `storm_id` 自动划分训练集和验证集，默认保留 20% 台风作为验证集，并保存验证集指标最好的权重。

训练完成后默认会生成：

```text
backend/models/weights/typhoon_pinn_v1.pth
backend/models/weights/typhoon_pinn_v1.summary.json
```

常用训练参数：

- `--val-ratio 0.2`：按台风分组的验证集比例，设为 `0` 可关闭验证集。
- `--patience 40`：早停轮数，连续若干轮没有提升就提前结束。
- `--device auto`：自动使用 `cuda` 或 `cpu`。
- `--report path\to\summary.json`：自定义训练摘要输出位置。
- `--velocity-weight`、`--inertia-weight`、`--coriolis-weight`、`--wind-pressure-weight`、`--nearshore-weight`：调节物理约束损失权重。

训练摘要 JSON 会记录：

- 数据集统计和训练/验证拆分结果。
- 每个 epoch 的 train/val loss。
- 最优 epoch 和最优监控指标。
- 本次训练使用的超参数和损失权重。

启动后端：

```powershell
python backend\app.py
```

启动前端：

```powershell
cd frontend
npm run dev
```

## 训练数据输入

训练数据支持 `.csv`、`.json`、`.jsonl`。推荐字段如下：

| 字段 | 是否必需 | 含义 |
| --- | --- | --- |
| `storm_id` | 否 | 台风编号；缺失时默认所有记录属于同一个台风 |
| `timestamp` | 与 `t_hours` 二选一 | 观测时间 |
| `t_hours` | 与 `timestamp` 二选一 | 相对该台风首个观测点的小时数 |
| `lng` | 是 | 经度 |
| `lat` | 是 | 纬度 |
| `wind_speed` | 是 | 最大风速，单位 m/s |
| `pressure` | 是 | 中心气压，单位 hPa |

训练建议：

- 数据按 `storm_id` 分组时，每个台风至少保留 `sequence_length + 1` 个点，默认至少需要 5 个点。
- 当前仓库中的 `CH2025BST_pinn_dataset.csv` 主要是 3 小时和 6 小时间隔；训练完成后做接口联调时，建议优先使用相近的 `time_step_hours`。
- 如果 CSV 来自 Excel 导出，脚本已经兼容 UTF-8 BOM，不需要手动去掉表头乱码。

## 接口输入

前端上传或后端接口 `POST /api/predict_typhoon` 接收台风观测 JSON。

| 字段 | 类型 | 是否必需 | 含义 |
| --- | --- | --- | --- |
| `storm_id` | string | 是 | 台风编号 |
| `storm_name` | string | 是 | 台风名称 |
| `basin` | string | 否 | 所属海域，默认 `East China Sea` |
| `forecast_steps` | number | 否 | 预测步数，范围 1 到 24 |
| `time_step_hours` | number | 否 | 每步预测时间间隔，范围 1 到 6 小时 |
| `observations` | array | 是 | 历史观测点，至少 2 个 |
| `actual_track` / `future_observations` | array | 否 | 真实未来轨迹，用于计算误差 |

每个观测点字段：

| 字段 | 类型 | 是否必需 | 含义 |
| --- | --- | --- | --- |
| `lng` | number | 是 | 经度 |
| `lat` | number | 是 | 纬度 |
| `timestamp` | string | 是 | UTC 时间戳 |
| `wind_speed` | number | 否 | 最大风速，单位 m/s |
| `pressure` | number | 否 | 中心气压，单位 hPa |

## 接口输出

`POST /api/predict_typhoon` 输出：

| 字段 | 含义 |
| --- | --- |
| `model_name` | 当前实际使用的模型，`pinn` 或 `rule_baseline` |
| `model_type` | 模型版本，当前为 `PINN-v1` |
| `observed_track` | 历史观测轨迹 |
| `predicted_track` | 当前主预测轨迹 |
| `pinn_track` | PINN 预测轨迹；无权重或推理失败时为空 |
| `baseline_track` | 线性平滑 baseline 轨迹 |
| `actual_track` | 可选真实未来轨迹 |
| `combined_track` | 历史轨迹 + 主预测轨迹 |
| `losses` | `data_loss`、`physics_loss`、速度一致性、惯性、科里奥利、风压关系、近岸衰减损失 |
| `metrics` | 轨迹 MAE、终点误差、风速 MAE、气压 MAE、baseline 对比指标 |
| `weather_context` | 生成气象场所需的预测台风中心、最大风速和中心气压 |
| `summary` | 输入点数、预测点数、模型名、推理模式、物理一致性得分等摘要 |

`GET /api/get_weather_conditions` 输出雨量、风速或气压网格数据，用于前端地图图层渲染。

## 无模型时的行为

如果缺少：

```text
backend/models/weights/typhoon_pinn_v1.pth
```

后端会自动使用 baseline 预测，响应中会出现：

```json
{
  "model_name": "rule_baseline",
  "model_type": "PINN-v1",
  "inference_mode": "linear_fallback"
}
```

这表示系统可以运行和展示，但还没有真正使用训练后的 PINN 模型。
