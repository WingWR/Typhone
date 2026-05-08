# TyphoonAI — 基于 PINN 的台风轨迹预测与可视化

Physics-Informed Neural Network 台风路径预测系统，React + DeckGL 前端 + Flask 后端，面向 AI for Science 气候建模方向。

## 快速启动

```powershell
# 一键启动 (Windows)
.\start.ps1
```

或分步启动：

```powershell
# 1. 安装依赖
pip install -r backend\requirements.txt
cd frontend && npm install && cd ..

# 2. 训练模型（如无预训练权重）
python train\train_pinn.py --dataset train\CH2025BST_pinn_dataset.csv --epochs 200

# 3. 启动服务
python backend\app.py          # → http://127.0.0.1:5000
cd frontend && npm run dev     # → http://127.0.0.1:5173
```

## 架构

```
train/                  backend/                frontend/
┌──────────────┐       ┌──────────────┐       ┌──────────────────┐
│ train_pinn.py│  →   │ Flask API    │  ←→   │ React + DeckGL   │
│ 训练数据集    │       │ /api/predict │       │ 地图 + 时间轴     │
└──────────────┘       │ /api/weather │       │ 侧栏 + 图例      │
                       └──────┬───────┘       └──────────────────┘
                              │
                       models/weights/
                       typhoon_pinn_v1.pth
```

- **训练模块** — 从台风观测 CSV/JSON 训练 PINN 模型，输出权重文件
- **后端** — Flask REST API，加载权重进行推理；无权重时自动降级为线性平滑外推
- **前端** — 上传观测数据，可视化预测轨迹，时间轴回放台风运动

## 训练

```powershell
python train\train_pinn.py --dataset train\CH2025BST_pinn_dataset.csv --epochs 200
```

产出：
- `backend/models/weights/typhoon_pinn_v1.pth` — 验证集最优权重
- `backend/models/weights/typhoon_pinn_v1.summary.json` — 每轮指标、超参、数据集统计

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 200 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--val-ratio` | 0.2 | 按台风划分的验证集比例，0 则关闭 |
| `--patience` | 40 | 早停容忍轮数，0 则关闭 |
| `--device` | auto | `auto` / `cuda` / `cpu` |
| `--sequence-length` | 4 | 输入序列长度 |
| `--hidden-dim` | 128 | 隐藏层维度 |

PINN 损失函数包含数据拟合项和五项物理约束：速度一致性、惯性约束、科里奥利力、风速-气压耦合、近岸衰减。

## 数据格式

### 训练数据（CSV / JSON / JSONL）

| 字段 | 必需 | 说明 |
|------|------|------|
| `storm_id` | 否 | 台风编号 |
| `timestamp` | 与 `t_hours` 二选一 | UTC 观测时间 |
| `t_hours` | 与 `timestamp` 二选一 | 距首个观测点的小时数 |
| `lng` | 是 | 经度 |
| `lat` | 是 | 纬度 |
| `wind_speed` | 是 | 最大风速 (m/s) |
| `pressure` | 是 | 中心气压 (hPa) |

每个台风至少 5 个观测点（sequence_length + 1）。CSV 含 UTF-8 BOM 会自动处理。

### 前端上传 JSON 格式

前端上传的 `.json` 文件必须包含 `storm_id`、`storm_name` 和 `observations` 三个字段：

```json
{
  "storm_id": "2026-DEMO",
  "storm_name": "Aster",
  "basin": "East China Sea",
  "forecast_steps": 8,
  "time_step_hours": 2,
  "observations": [
    {"lng": 124.18, "lat": 26.54, "timestamp": "2026-07-21T00:00:00Z", "wind_speed": 33, "pressure": 980},
    {"lng": 123.90, "lat": 27.00, "timestamp": "2026-07-21T02:00:00Z", "wind_speed": 35, "pressure": 976},
    {"lng": 123.55, "lat": 27.58, "timestamp": "2026-07-21T04:00:00Z", "wind_speed": 38, "pressure": 970},
    {"lng": 123.10, "lat": 28.20, "timestamp": "2026-07-21T06:00:00Z", "wind_speed": 42, "pressure": 962},
    {"lng": 122.65, "lat": 28.85, "timestamp": "2026-07-21T08:00:00Z", "wind_speed": 45, "pressure": 955}
  ]
}
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `storm_id` | 是 | 台风编号 |
| `storm_name` | 是 | 台风名称 |
| `basin` | 否 | 所在海域，默认 `East China Sea` |
| `forecast_steps` | 否 | 预测步数，默认 8，最大 24 |
| `time_step_hours` | 否 | 每步小时数，默认 2，最大 6 |
| `observations` | 是 | 观测点数组，至少 2 个，按时间升序 |
| `observations[].lng` | 是 | 经度 |
| `observations[].lat` | 是 | 纬度 |
| `observations[].timestamp` | 是 | UTC 时间 (ISO 8601) |
| `observations[].wind_speed` | 否 | 最大风速 (m/s)，默认 25 |
| `observations[].pressure` | 否 | 中心气压 (hPa)，默认 990 |

上传不符合格式的 JSON 文件时，页面会弹出错误提示。

## API

基础地址：`http://127.0.0.1:5000/api`

### `GET /api/health`
```json
{"status": "ok"}
```

### `POST /api/predict_typhoon`
提交观测数据，获取预测轨迹。

**请求体：**
```json
{
  "storm_id": "2026-DEMO",
  "storm_name": "Aster",
  "basin": "East China Sea",
  "forecast_steps": 8,
  "time_step_hours": 2,
  "observations": [
    {"lng": 124.18, "lat": 26.54, "timestamp": "2026-07-21T00:00:00Z", "wind_speed": 33, "pressure": 980},
    {"lng": 123.90, "lat": 27.00, "timestamp": "2026-07-21T02:00:00Z", "wind_speed": 35, "pressure": 976}
  ]
}
```

**主要返回字段：**

| 字段 | 说明 |
|------|------|
| `combined_track` | 观测 + 预测完整轨迹 |
| `pinn_track` | PINN 预测轨迹（降级时为空） |
| `baseline_track` | 线性平滑基线轨迹 |
| `losses` | 六项物理损失分解 |
| `metrics` | 轨迹、风速、气压误差 |
| `summary.inference_mode` | `"pinn"` 或 `"linear_fallback"` |
| `weather_context` | 气象图层所需的台风中心参数 |

### `GET /api/get_weather_conditions`
返回雨量/风速/气压网格数据。Query 参数：`field`、`center_lng`、`center_lat`、`max_wind_speed`、`central_pressure`。

## 降级行为

当 `backend/models/weights/typhoon_pinn_v1.pth` 不存在时，接口自动切换为线性平滑基线预测。系统仍然可用，但 `inference_mode` 会返回 `"linear_fallback"`。训练模型后即可启用完整 PINN 推理。

## 项目结构

```
Typhone/
├── start.ps1                    # 一键启动脚本
├── train/
│   ├── train_pinn.py            # 训练管线
│   ├── CH2025BST_pinn_dataset.csv
│   └── README.md                # 训练说明
├── backend/
│   ├── app.py                   # Flask 入口
│   ├── config.py                # 域参数配置
│   ├── requirements.txt
│   ├── api/routes.py            # REST 路由
│   ├── logic/pinn_inference.py  # 推理与降级逻辑
│   ├── models/pinn_model.py     # PyTorch 模型定义
│   ├── models/weights/          # 权重存储
│   ├── services/weather_service.py
│   ├── utils/physics_engine.py  # 物理工具
│   ├── utils/request_parser.py
│   └── utils/validation.py
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── hooks/useTyphoonVisualizer.js
    │   ├── components/
    │   │   ├── MapScene.jsx          # DeckGL 地图
    │   │   ├── ForecastSidebar.jsx   # 可折叠侧栏
    │   │   ├── MapLegend.jsx         # 图例 + 当前点数据
    │   │   └── TimelineBar.jsx       # 播放时间轴
    │   ├── api/typhoonApi.js
    │   ├── utils/track.js
    │   └── constants/map.js
    └── package.json
```

## 技术栈

| 层 | 技术 |
|----|------|
| 模型 | PyTorch, PINN（物理约束损失） |
| 后端 | Flask, NumPy, Pandas |
| 前端 | React 19, DeckGL 9, MUI 7, Vite |
