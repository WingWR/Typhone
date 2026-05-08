# PINN Training Data Format

训练入口：

```powershell
python train/train_pinn.py --dataset path\to\typhoon_dataset.csv --epochs 200 --batch-size 32 --sequence-length 4 --hidden-dim 128 --lr 1e-3
```

默认行为：

- 按 `storm_id` 划分训练集和验证集，默认 `--val-ratio 0.2`。
- 自动保存验证集最优权重，而不是最后一轮权重。
- 默认输出同名训练摘要 JSON，便于写实验记录和报告。

默认输出：

```text
backend/models/weights/typhoon_pinn_v1.pth
backend/models/weights/typhoon_pinn_v1.summary.json
```

常用参数：

- `--val-ratio 0.2`：验证集比例，按台风分组拆分。
- `--patience 40`：早停轮数。
- `--device auto`：自动选择 GPU 或 CPU。
- `--report path\to\summary.json`：自定义训练摘要位置。
- `--velocity-weight`、`--inertia-weight`、`--coriolis-weight`、`--wind-pressure-weight`、`--nearshore-weight`：物理损失权重。

训练数据支持 `.csv`、`.json`、`.jsonl`，推荐字段如下：

| column | required | description |
| --- | --- | --- |
| `storm_id` | no | 台风编号；缺失时默认所有记录属于同一个台风 |
| `timestamp` | yes, if no `t_hours` | 观测时间 |
| `t_hours` | yes, if no `timestamp` | 相对该台风首个观测点的小时数 |
| `lng` | yes | 经度 |
| `lat` | yes | 纬度 |
| `wind_speed` | yes | 最大风速，单位 m/s |
| `pressure` | yes | 中心气压，单位 hPa |

补充说明：

- CSV 读取兼容 UTF-8 BOM。
- `.jsonl` 会按逐行 JSON 读取。
- 每个台风至少需要 `sequence_length + 1` 个点才能产出训练样本。
- 如果训练数据主要是 3 小时或 6 小时间隔，联调接口时建议使用接近的 `time_step_hours`。
