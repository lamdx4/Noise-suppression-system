"""
RNNoise Training Logger - JSON output cho báo cáo
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List
import torch


class TrainingLogger:
    """Logger ghi training metrics ra JSON file cho báo cáo."""
    
    def __init__(self, log_dir: str, experiment_name: str = "rnnoise"):
        """
        Args:
            log_dir: Thư mục lưu log files
            experiment_name: Tên experiment (dùng cho filename)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Timestamp cho experiment này
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log file paths
        self.metrics_file = os.path.join(
            log_dir, 
            f"{experiment_name}_{self.timestamp}_metrics.json"
        )
        self.config_file = os.path.join(
            log_dir,
            f"{experiment_name}_{self.timestamp}_config.json"
        )
        self.summary_file = os.path.join(
            log_dir,
            f"{experiment_name}_{self.timestamp}_summary.json"
        )
        
        # Data storage
        self.epoch_metrics: List[Dict] = []
        self.batch_metrics: List[Dict] = []
        self.config: Dict = {}
        
    def log_config(self, config: Dict[str, Any]):
        """Lưu training configuration."""
        self.config = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "config": config
        }
        
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def log_epoch(self, 
                  epoch: int,
                  train_loss: float,
                  train_gain_loss: float,
                  train_vad_loss: float,
                  learning_rate: float,
                  **kwargs):
        """
        Ghi metrics sau mỗi epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Total training loss
            train_gain_loss: Gain prediction loss
            train_vad_loss: VAD prediction loss  
            learning_rate: Current learning rate
            **kwargs: Additional metrics (validation, etc.)
        """
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train": {
                "loss": float(train_loss),
                "gain_loss": float(train_gain_loss),
                "vad_loss": float(train_vad_loss)
            },
            "learning_rate": float(learning_rate),
            **kwargs
        }
        
        self.epoch_metrics.append(epoch_data)
        
        # Ghi incremental (mỗi epoch ghi luôn)
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                "experiment": self.experiment_name,
                "start_time": self.timestamp,
                "epochs": self.epoch_metrics
            }, f, indent=2, ensure_ascii=False)
    
    def log_batch(self,
                  epoch: int,
                  batch_idx: int,
                  loss: float,
                  gain_loss: float,
                  vad_loss: float):
        """
        Ghi metrics từng batch (optional - cho detailed analysis).
        
        Args:
            epoch: Current epoch
            batch_idx: Batch index
            loss: Batch loss
            gain_loss: Batch gain loss
            vad_loss: Batch VAD loss
        """
        batch_data = {
            "epoch": epoch,
            "batch": batch_idx,
            "loss": float(loss),
            "gain_loss": float(gain_loss),
            "vad_loss": float(vad_loss)
        }
        
        self.batch_metrics.append(batch_data)
    
    def save_summary(self, 
                     total_epochs: int,
                     best_epoch: int,
                     best_loss: float,
                     final_model_path: str,
                     **kwargs):
        """
        Lưu summary sau khi training xong.
        
        Args:
            total_epochs: Tổng số epochs trained
            best_epoch: Epoch có loss tốt nhất
            best_loss: Best loss achieved
            final_model_path: Path to final model
            **kwargs: Additional summary info
        """
        # Tính statistics
        losses = [epoch["train"]["loss"] for epoch in self.epoch_metrics]
        
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "training_duration": {
                "start": self.epoch_metrics[0]["timestamp"] if self.epoch_metrics else None,
                "end": self.epoch_metrics[-1]["timestamp"] if self.epoch_metrics else None,
            },
            "training_stats": {
                "total_epochs": total_epochs,
                "best_epoch": best_epoch,
                "best_loss": float(best_loss),
                "final_loss": float(losses[-1]) if losses else None,
                "avg_loss": float(sum(losses) / len(losses)) if losses else None,
                "min_loss": float(min(losses)) if losses else None,
            },
            "model": {
                "final_checkpoint": final_model_path,
                "best_checkpoint": kwargs.get("best_model_path", "")
            },
            "config": self.config.get("config", {}),
            **kwargs
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Logs saved:")
        print(f"  - Config: {self.config_file}")
        print(f"  - Metrics: {self.metrics_file}")
        print(f"  - Summary: {self.summary_file}")
    
    def get_latest_metrics(self) -> Dict:
        """Lấy metrics của epoch cuối cùng."""
        if self.epoch_metrics:
            return self.epoch_metrics[-1]
        return {}


# === Usage Example ===
def example_usage():
    """Example cách dùng logger."""
    
    # 1. Khởi tạo logger
    logger = TrainingLogger(
        log_dir="ai/logs",
        experiment_name="rnnoise_sparse_384"
    )
    
    # 2. Log config trước khi train
    logger.log_config({
        "model": {
            "gru_size": 384,
            "cond_size": 128,
            "input_dim": 42,
            "output_dim": 22
        },
        "training": {
            "epochs": 150,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "sparse": True
        },
        "data": {
            "features_file": "features.f32",
            "num_sequences": 30000
        }
    })
    
    # 3. Training loop
    for epoch in range(1, 151):
        # ... training code ...
        
        # Log mỗi epoch
        logger.log_epoch(
            epoch=epoch,
            train_loss=0.0123,  # Replace with actual
            train_gain_loss=0.0112,
            train_vad_loss=0.0011,
            learning_rate=1e-3 / (1 + epoch * 5e-5)
        )
    
    # 4. Sau khi train xong
    logger.save_summary(
        total_epochs=150,
        best_epoch=145,
        best_loss=0.0098,
        final_model_path="ai/models/rnnoise_150.pth",
        best_model_path="ai/models/rnnoise_best.pth",
        # Additional info
        hardware="NVIDIA RTX 3060",
        training_time_hours=6.5
    )


if __name__ == "__main__":
    example_usage()
