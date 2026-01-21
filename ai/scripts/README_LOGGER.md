# Training Logger cho RNNoise

Module logging cho training, output JSON format để viết báo cáo.

## Sử dụng

```python
from training_logger import TrainingLogger

# 1. Khởi tạo
logger = TrainingLogger(
    log_dir="ai/logs",
    experiment_name="rnnoise_sparse_384"
)

# 2. Log config
logger.log_config({
    "model": {"gru_size": 384, ...},
    "training": {"epochs": 150, ...}
})

# 3. Log trong training loop
for epoch in range(epochs):
    # ... training ...
    logger.log_epoch(
        epoch=epoch,
        train_loss=loss,
        train_gain_loss=gain_loss,
        train_vad_loss=vad_loss,
        learning_rate=current_lr
    )

# 4. Sau training
logger.save_summary(
    total_epochs=150,
    best_epoch=145,
    best_loss=0.0098,
    final_model_path="models/rnnoise_150.pth"
)
```

## Output Files

Logger tạo 3 files JSON:

### 1. Config (`*_config.json`)

```json
{
  "experiment_name": "rnnoise_sparse_384",
  "timestamp": "20260121_143000",
  "config": {
    "model": {"gru_size": 384, ...},
    "training": {"epochs": 150, ...}
  }
}
```

### 2. Metrics (`*_metrics.json`)

```json
{
  "experiment": "rnnoise_sparse_384",
  "start_time": "20260121_143000",
  "epochs": [
    {
      "epoch": 1,
      "timestamp": "2026-01-21T14:30:15",
      "train": {
        "loss": 0.1234,
        "gain_loss": 0.1123,
        "vad_loss": 0.0111
      },
      "learning_rate": 0.001
    },
    ...
  ]
}
```

### 3. Summary (`*_summary.json`)

```json
{
  "experiment_name": "rnnoise_sparse_384",
  "training_stats": {
    "total_epochs": 150,
    "best_epoch": 145,
    "best_loss": 0.0098,
    "final_loss": 0.0102,
    "avg_loss": 0.0234
  },
  "model": {
    "final_checkpoint": "models/rnnoise_150.pth"
  }
}
```

## Dùng cho Báo Cáo

```python
import json

# Đọc summary
with open("ai/logs/rnnoise_sparse_384_*_summary.json") as f:
    summary = json.load(f)

print(f"Best loss: {summary['training_stats']['best_loss']}")
print(f"Total epochs: {summary['training_stats']['total_epochs']}")

# Đọc metrics để vẽ charts
with open("ai/logs/rnnoise_sparse_384_*_metrics.json") as f:
    metrics = json.load(f)

losses = [epoch['train']['loss'] for epoch in metrics['epochs']]
# Plot losses...
```

## Features

- ✅ JSON format (dễ parse cho báo cáo)
- ✅ Auto timestamp
- ✅ Incremental save (mỗi epoch ghi luôn)
- ✅ Summary statistics
- ✅ Tiếng Việt friendly
- ✅ Lightweight (không dependencies nhiều)
