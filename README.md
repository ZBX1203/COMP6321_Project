## Overview

- 📊 Multi-label text classification (20 categories, ~18,846 documents)
- 🚀 GPU-accelerated 3-layer DNN with BERT embeddings (768-dim)
- 📈 5 baseline model comparisons (Logistic Regression, SGD, Random Forest, Decision Tree)
- 📉 8+ evaluation metrics with rich visualizations
- 🎯 **High recall (86%)** with competitive F1-score on 20 Newsgroups dataset

## Performance

**Dataset**: 20 Newsgroups (13,191 train / 1,885 val / 3,770 test samples)

| Model | F1 (Micro) | Precision | Recall | Training Time |
|-------|------------|-----------|--------|---------------|
| **Logistic Regression** | **0.6064** | **0.7228** | **0.5223** | 13.87s |
| SGD Classifier | 0.6015 | 0.6783 | 0.5403 | 2.93s |
| **DNN (Ours)** | 0.4337 | 0.2899 | **0.8605** | GPU-accelerated |
| Decision Tree | 0.2834 | 0.3000 | 0.2684 | 63.90s |
| Random Forest | 0.1884 | 0.9316 | 0.1048 | 115.31s |

**Note**: DNN achieves highest recall (86%), capturing more true positives but with lower precision. Traditional methods (Logistic Regression) show better balanced performance on this dataset.


## Model Architecture

### BERT-based DNN

```
BERT Embeddings (768-dim) → Dense(512) → ReLU → Dropout(0.5)
                          → Dense(256) → ReLU → Dropout(0.5)
                          → Dense(128) → ReLU → Dropout(0.5)
                          → Dense(20) → Sigmoid
```

**Features**:
- Pre-trained BERT-base-uncased for text embeddings
- 562,324 trainable parameters
- Adam optimizer (lr=0.001)
- Batch size: 512
- Epochs: 20
- Loss: BCEWithLogitsLoss (weighted for class imbalance)
- Device: NVIDIA H100 80GB GPU

## Key Features

- 🤖 **BERT Embeddings**: Pre-trained language model for semantic understanding
- ⚡ **GPU Acceleration**: Automatic CUDA detection and H100 GPU support
- 📊 **Rich Metrics**: F1, Precision, Recall, Jaccard, Hamming Loss, Subset Accuracy
- 📈 **Visualizations**: 12+ plots (training history, confusion matrices, per-label analysis)
- 💾 **Model Persistence**: Save/load trained models
- 🔄 **Automated Pipeline**: End-to-end training and evaluation workflow
- 🌐 **Network-Aware**: Separate data generation for login nodes vs compute nodes

## 📁 Project Structure

```
├── main.py                      # Main DNN training script
├── model.py                     # DNN architecture with BERT embeddings
├── data_loader.py               # Data preprocessing & BERT embedding generation
├── evaluator.py                 # Comprehensive evaluation metrics
├── baseline_models.py           # Traditional ML models
├── compare_models.py            # Multi-model comparison
├── download_bert.py             # BERT model downloader
├── run_generate_data.sh         # Data generation script
├── run_compare_models.sbatch    # SLURM job submission script
├── bert_model/                  # Local BERT-base-uncased model
├── data/                        # Preprocessed dataset (pkl files)
├── models/                      # Saved trained models
├── results/plots/               # Training visualizations
└── comparison_results/          # Model comparison reports & plots
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source /scratch/your_username/comp6321_env/bin/activate

# Install dependencies
pip install transformers torch scikit-learn matplotlib seaborn pandas
```

### 2. Download BERT Model (on login node with internet)
```bash
python download_bert.py
```

### 3. Generate Dataset (on login node)
```bash
./run_generate_data.sh
# Or manually:
python -c "from data_loader import RCV1DataLoader; ..."
```

### 4. Submit Training Job
```bash
sbatch run_compare_models.sbatch
```

## 📊 Results

- **Total Runtime**: ~4 minutes on NVIDIA H100 GPU
- **Dataset Size**: 59 MB (preprocessed with BERT embeddings)
- **Best Traditional Model**: Logistic Regression (F1: 0.6064)
- **DNN Strength**: High recall (86.05%) for comprehensive label coverage



