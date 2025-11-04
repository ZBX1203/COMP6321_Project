
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    hamming_loss, accuracy_score, jaccard_score
)
from data_loader import RCV1DataLoader
from model import DNNClassifier
from baseline_models import BaselineModel, DummyBaseline
from evaluator import MultiLabelEvaluator
import os
import time
import json


class ModelComparator:
    
    def __init__(self, output_dir='comparison_results'):
        """Initialize comparator"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        self.models = {}
        self.results = []
        
    def add_model(self, name, model, skip_training=False):
        """
        Args:
            name: Model name
            model: Model object
            skip_training: If True, don't train this model (already trained)
        """
        self.models[name] = {'model': model, 'skip_training': skip_training}
        print(f"✓ Added model: {name}" + (" (pre-trained)" if skip_training else ""))
    
    def load_data(self, max_train_samples=None):
        """Load data for comparison"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        data_loader = RCV1DataLoader(data_dir='data')
        
        X_train, y_train = data_loader.load_data('train')
        X_val, y_val = data_loader.load_data('val')
        X_test, y_test = data_loader.load_data('test')
        metadata = data_loader.load_metadata()
        
        if max_train_samples and X_train.shape[0] > max_train_samples:
            print(f"Sampling {max_train_samples:,} training samples...")
            indices = np.random.choice(X_train.shape[0], max_train_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = metadata['target_names']
        
        print(f"✓ Loaded data:")
        print(f"  Training:   {X_train.shape[0]:,} samples")
        print(f"  Validation: {X_val.shape[0]:,} samples")
        print(f"  Test:       {X_test.shape[0]:,} samples")
    
    def train_all_models(self):
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            skip_training = model_dict['skip_training']
            
            if skip_training:
                print(f"\n{'─'*70}")
                print(f"Skipping training: {name} (already trained)")
                print(f"{'─'*70}")
                continue
            
            print(f"\n{'─'*70}")
            print(f"Training: {name}")
            print(f"{'─'*70}")
            
            try:
                model.fit(self.X_train, self.y_train)
                
                model_path = os.path.join(self.output_dir, f'{name}_model.pkl')
                model.save(model_path)
                
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
                import traceback
                traceback.print_exc()
    
    def evaluate_all_models(self):
        print("\n" + "="*70)
        print("EVALUATING ALL MODELS")
        print("="*70)
        
        self.results = []
        
        for name, model_dict in self.models.items():
            model = model_dict['model']
            
            print(f"\n{'─'*70}")
            print(f"Evaluating: {name}")
            print(f"{'─'*70}")
            
            try:
                if hasattr(model, 'predict'):
                    if isinstance(model, (BaselineModel, DummyBaseline)):
                        y_pred, pred_time = model.predict(self.X_test)
                    else:  # DNN
                        start_time = time.time()
                        y_pred = model.predict(self.X_test)
                        pred_time = time.time() - start_time
                else:
                    print(f"✗ {name} does not have predict method")
                    continue
                
                if hasattr(self.y_test, 'toarray'):
                    y_test = self.y_test.toarray()
                else:
                    y_test = np.asarray(self.y_test)
                
                y_pred = np.asarray(y_pred)
                
                metrics = {
                    'model_name': name,
                    'training_time': getattr(model, 'training_time', 0),
                    'prediction_time': pred_time,
                    'hamming_loss': hamming_loss(y_test, y_pred),
                    'subset_accuracy': accuracy_score(y_test, y_pred),
                    'jaccard_score': jaccard_score(y_test, y_pred, average='samples', zero_division=0),
                    'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
                    'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                    'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
                    'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                    'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
                    'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                }
                
                self.results.append(metrics)
                
                # Print metrics
                print(f"\nMetrics for {name}:")
                print(f"  F1 (Micro):      {metrics['f1_micro']:.4f}")
                print(f"  F1 (Macro):      {metrics['f1_macro']:.4f}")
                print(f"  Precision:       {metrics['precision_micro']:.4f}")
                print(f"  Recall:          {metrics['recall_micro']:.4f}")
                print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
                print(f"  Training Time:   {metrics['training_time']:.2f}s")
                print(f"  Prediction Time: {metrics['prediction_time']:.2f}s")
                
            except Exception as e:
                print(f"✗ Failed to evaluate {name}: {e}")
                import traceback
                traceback.print_exc()
    
    def save_results(self):
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        if not self.results:
            print("⚠ No results to save")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = os.path.join(self.output_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV: {csv_path}")
        
        json_path = os.path.join(self.output_dir, 'model_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
        
        txt_path = os.path.join(self.output_dir, 'model_comparison_report.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            sorted_results = sorted(self.results, key=lambda x: x['f1_micro'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"\n{i}. {result['model_name']}\n")
                f.write("-"*70 + "\n")
                f.write(f"F1 (Micro):       {result['f1_micro']:.4f}\n")
                f.write(f"F1 (Macro):       {result['f1_macro']:.4f}\n")
                f.write(f"F1 (Weighted):    {result['f1_weighted']:.4f}\n")
                f.write(f"Precision (Micro): {result['precision_micro']:.4f}\n")
                f.write(f"Recall (Micro):    {result['recall_micro']:.4f}\n")
                f.write(f"Subset Accuracy:   {result['subset_accuracy']:.4f}\n")
                f.write(f"Jaccard Score:     {result['jaccard_score']:.4f}\n")
                f.write(f"Hamming Loss:      {result['hamming_loss']:.4f}\n")
                f.write(f"Training Time:     {result['training_time']:.2f}s\n")
                f.write(f"Prediction Time:   {result['prediction_time']:.2f}s\n")
        
        print(f"✓ Saved report: {txt_path}")
    
    def plot_comparison(self):
        if not self.results:
            print("⚠ No results to plot")
            return
            
        print("\n" + "="*70)
        print("GENERATING COMPARISON PLOTS")
        print("="*70)
        
        df = pd.DataFrame(self.results)
        
        # 1. F1 Scores Comparison
        self._plot_f1_comparison(df)
        
        # 2. All Metrics Heatmap
        self._plot_metrics_heatmap(df)
        
        # 3. Training Time vs Performance
        self._plot_time_vs_performance(df)
        
        # 4. Precision-Recall Trade-off
        self._plot_precision_recall(df)
        
        print("✓ All comparison plots generated")
    
    def _plot_f1_comparison(self, df):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['f1_micro'], width, label='F1 Micro', alpha=0.8, color='#3498db')
        ax.bar(x, df['f1_macro'], width, label='F1 Macro', alpha=0.8, color='#e74c3c')
        ax.bar(x + width, df['f1_weighted'], width, label='F1 Weighted', alpha=0.8, color='#2ecc71')
        
        ax.set_xlabel('Model', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
        ax.set_title('F1 Score Comparison Across Models', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['model_name'], rotation=45, ha='right', fontsize=11)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'f1_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ F1 comparison plot")
    
    def _plot_metrics_heatmap(self, df):
        metrics_cols = ['f1_micro', 'f1_macro', 'precision_micro', 'recall_micro', 
                       'subset_accuracy', 'jaccard_score']
        
        data = df[metrics_cols].values.T
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   xticklabels=df['model_name'],
                   yticklabels=['F1 Micro', 'F1 Macro', 'Precision', 'Recall', 
                               'Subset Acc', 'Jaccard'],
                   cbar_kws={'label': 'Score'},
                   linewidths=1, linecolor='gray', ax=ax,
                   vmin=0, vmax=1)
        
        ax.set_title('Model Performance Heatmap', fontsize=15, fontweight='bold', pad=20)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'metrics_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Metrics heatmap")
    
    def _plot_time_vs_performance(self, df):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training time vs F1
        ax1.scatter(df['training_time'], df['f1_micro'], s=150, alpha=0.7, c=df['f1_micro'], cmap='viridis')
        for i, name in enumerate(df['model_name']):
            ax1.annotate(name, (df['training_time'].iloc[i], df['f1_micro'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1 Score (Micro)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Time vs Performance', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.set_xscale('log')
        
        # Prediction time vs F1
        ax2.scatter(df['prediction_time'], df['f1_micro'], s=150, alpha=0.7, c=df['f1_micro'], cmap='plasma')
        for i, name in enumerate(df['model_name']):
            ax2.annotate(name, (df['prediction_time'].iloc[i], df['f1_micro'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Prediction Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score (Micro)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Time vs Performance', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'time_vs_performance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Time vs performance plot")
    
    def _plot_precision_recall(self, df):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(df['recall_micro'], df['precision_micro'], 
                           s=200, alpha=0.7, c=df['f1_micro'], 
                           cmap='RdYlGn', edgecolors='black', linewidth=2)
        
        for i, name in enumerate(df['model_name']):
            ax.annotate(name, 
                       (df['recall_micro'].iloc[i], df['precision_micro'].iloc[i]),
                       xytext=(7, 7), textcoords='offset points', 
                       fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Recall (Micro)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision (Micro)', fontsize=13, fontweight='bold')
        ax.set_title('Precision-Recall Trade-off', fontsize=15, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])
        
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, linewidth=2, label='Equal P/R')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Score (Micro)', fontsize=11, fontweight='bold')
        
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'precision_recall.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Precision-recall plot")
    
    def print_summary(self):
        if not self.results:
            print("⚠ No results to summarize")
            return
            
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        df = pd.DataFrame(self.results)
        df_sorted = df.sort_values('f1_micro', ascending=False)
        
        print("\nRanking by F1 Score (Micro):")
        print("-"*70)
        
        for idx, (i, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"{idx}. {row['model_name']:25s} "
                  f"F1: {row['f1_micro']:.4f}  "
                  f"P: {row['precision_micro']:.4f}  "
                  f"R: {row['recall_micro']:.4f}  "
                  f"Time: {row['training_time']:7.2f}s")
        
        print("\n" + "="*70)


def main():
    
    print("\n" + "="*70)
    print(" COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    comparator = ModelComparator(output_dir='comparison_results')
    
    comparator.load_data(max_train_samples=50000)  # Use 50k for faster training
    
    print("\n" + "="*70)
    print("ADDING MODELS FOR COMPARISON")
    print("="*70)
    

    comparator.add_model('Logistic-Regression', 
                        BaselineModel('logistic', max_iter=1000, n_jobs=-1))
    
    comparator.add_model('SGD-Classifier', 
                        BaselineModel('sgd', max_iter=1000, n_jobs=-1))
    
    comparator.add_model('Random-Forest', 
                        BaselineModel('rf', n_estimators=100, max_depth=20, n_jobs=-1))
    
    comparator.add_model('Naive-Bayes', 
                        BaselineModel('nb', alpha=1.0, n_jobs=-1))
    
    comparator.add_model('Decision-Tree', 
                        BaselineModel('dt', max_depth=20, n_jobs=-1))
    
    try:
        dnn_model = DNNClassifier.load('models/dnn_model.pth')
        comparator.add_model('DNN (Our Model)', dnn_model, skip_training=True)
    except:
        print("⚠ DNN model not found. Train it first with main.py")

    comparator.train_all_models()
    
    comparator.evaluate_all_models()
    
    comparator.save_results()
    
    comparator.plot_comparison()
    
    comparator.print_summary()
    
    print("\n" + "="*70)
    print(" COMPARISON COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: comparison_results/")
    print("  - model_comparison.csv")
    print("  - model_comparison.json")
    print("  - model_comparison_report.txt")
    print("  - plots/")
    print("\n")


if __name__ == '__main__':
    main()