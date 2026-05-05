# PINN Training Data Format

训练入口：

```powershell
python train/train_pinn.py --dataset path\to\typhoon_dataset.csv --epochs 200
```

兼容入口：

```powershell
python backend/training/train_pinn.py --dataset path\to\typhoon_dataset.csv --epochs 200
```

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

训练完成后默认输出：

```text
backend/models/weights/typhoon_pinn_v1.pth
```
