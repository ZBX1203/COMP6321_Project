"""快速测试脚本 - 验证代码能否正常运行"""

from data_loader import RCV1DataLoader
from model import DNNClassifier
import numpy as np
import torch

print("="*70)
print("QUICK TEST - Validating Code")
print("="*70)

# 测试 GPU 检测
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 测试数据加载
print("\n1. Testing data loader...")
data_loader = RCV1DataLoader(data_dir='/scratch/bingxu97/data_test')
X_train, y_train, X_val, y_val, X_test, y_test, target_names = \
    data_loader.download_and_split(
        test_size=0.2,
        val_size=0.1,
        sample_size=5000,  # 只用 5000 样本快速测试
        full=False
    )
print(f"✓ Data loaded: {X_train.shape[0]} samples")

# 测试模型训练
print("\n2. Testing model training...")
model = DNNClassifier(
    hidden_layers=[128, 64],  # 更小的网络
    activation='relu',
    dropout=0.3,
    learning_rate=0.001,
    batch_size=128,
    epochs=2,  # 只训练 2 个 epoch
    device='cpu',  # 本地用 CPU
    random_state=42
)

history = model.fit(X_train, y_train, X_val, y_val)
print("✓ Training completed")

# 测试预测
print("\n3. Testing prediction...")
y_pred = model.predict(X_test[:100])
print(f"✓ Prediction shape: {y_pred.shape}")

print("\n" + "="*70)
print("ALL TESTS PASSED! Code is ready for Trillium.")
print("="*70)
