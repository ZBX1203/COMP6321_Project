
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    hamming_loss, accuracy_score, precision_score, 
    recall_score, f1_score, multilabel_confusion_matrix, jaccard_score
)
from scipy.sparse import issparse
import os


class MultiLabelEvaluator:
    
    def __init__(self, output_dir='results/plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def _to_numpy(self, data):
        if issparse(data):
            return data.toarray()
        elif isinstance(data, np.matrix):
            return np.asarray(data)
        else:
            return np.asarray(data)
    
    def evaluate(self, y_true, y_pred, target_names=None, dataset_name='Test'):
        # Ensure NumPy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        print("\n" + "="*70)
        print(f"EVALUATION RESULTS - {dataset_name.upper()} SET")
        print("="*70)
        
        # Calculate comprehensive metrics
        metrics = {
            'hamming_loss': hamming_loss(y_true, y_pred),
            'subset_accuracy': accuracy_score(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
            
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_samples': precision_score(y_true, y_pred, average='samples', zero_division=0),
            
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_samples': recall_score(y_true, y_pred, average='samples', zero_division=0),
            
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
        }
        
        true_labels_per_sample = np.sum(y_true, axis=1)
        pred_labels_per_sample = np.sum(y_pred, axis=1)
        metrics['avg_true_labels'] = float(np.mean(true_labels_per_sample))
        metrics['avg_pred_labels'] = float(np.mean(pred_labels_per_sample))
        
        print(f"\n{'─'*70}")
        print("OVERALL METRICS")
        print(f"{'─'*70}")
        print(f"Hamming Loss:     {metrics['hamming_loss']:.4f}")
        print(f"Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
        print(f"Jaccard Score:    {metrics['jaccard_score']:.4f}")
        
        print(f"\n{'─'*70}")
        print("MICRO-AVERAGED METRICS")
        print(f"{'─'*70}")
        print(f"Precision: {metrics['precision_micro']:.4f}")
        print(f"Recall:    {metrics['recall_micro']:.4f}")
        print(f"F1-Score:  {metrics['f1_micro']:.4f}")
        
        print(f"\n{'─'*70}")
        print("MACRO-AVERAGED METRICS")
        print(f"{'─'*70}")
        print(f"Precision: {metrics['precision_macro']:.4f}")
        print(f"Recall:    {metrics['recall_macro']:.4f}")
        print(f"F1-Score:  {metrics['f1_macro']:.4f}")
        
        print(f"\n{'─'*70}")
        print("LABEL STATISTICS")
        print(f"{'─'*70}")
        print(f"Avg true labels per sample:      {metrics['avg_true_labels']:.2f}")
        print(f"Avg predicted labels per sample: {metrics['avg_pred_labels']:.2f}")
        
        return metrics
    
    def print_sample_predictions(self, X, y_true, y_pred, target_names=None, n_samples=10):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        print("\n" + "="*70)
        print("SAMPLE PREDICTIONS")
        print("="*70)
        
        n_samples = min(n_samples, y_true.shape[0])
        
        for i in range(n_samples):
            print(f"\n{'─'*70}")
            print(f"Test Case {i+1}:")
            print(f"{'─'*70}")
            
            true_indices = np.where(y_true[i] == 1)[0]
            pred_indices = np.where(y_pred[i] == 1)[0]
            
            if target_names is not None:
                true_labels = [target_names[idx] for idx in true_indices]
                pred_labels = [target_names[idx] for idx in pred_indices]
            else:
                true_labels = [f"Label_{idx}" for idx in true_indices]
                pred_labels = [f"Label_{idx}" for idx in pred_indices]
            
            print(f"Ground Truth (y):  {len(true_labels)} labels")
            if len(true_labels) > 0:
                for label in true_labels:
                    print(f"  • {label}")
            else:
                print("  (No labels)")
            
            print(f"\nPredicted (y'):    {len(pred_labels)} labels")
            if len(pred_labels) > 0:
                for label in pred_labels:
                    print(f"  • {label}")
            else:
                print("  (No labels)")
            
            correct = np.sum(y_true[i] == y_pred[i])
            total = y_true.shape[1]
            accuracy = correct / total
            
            tp = np.sum((y_true[i] == 1) & (y_pred[i] == 1))
            fp = np.sum((y_true[i] == 0) & (y_pred[i] == 1))
            fn = np.sum((y_true[i] == 1) & (y_pred[i] == 0))
            
            print(f"\nSample Metrics:")
            print(f"  Accuracy:        {accuracy:.2%} ({correct}/{total} labels correct)")
            print(f"  True Positives:  {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            
            if tp == len(true_labels) and fp == 0 and fn == 0:
                print(f"  Status: ✓ PERFECT MATCH")
            elif tp > 0:
                print(f"  Status: ⚠ PARTIAL MATCH")
            else:
                print(f"  Status: ✗ NO MATCH")
        
        print("\n" + "="*70)
    
    def print_detailed_predictions(self, X, y_true, y_pred, target_names=None, 
                                   sample_indices=None):
        """
        Print detailed predictions for specific samples
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        if sample_indices is None:
            sample_indices = range(min(5, y_true.shape[0]))
        
        print("\n" + "="*70)
        print("DETAILED PREDICTION ANALYSIS")
        print("="*70)
        
        for idx in sample_indices:
            if idx >= y_true.shape[0]:
                continue
                
            print(f"\n{'═'*70}")
            print(f"TEST CASE #{idx}")
            print(f"{'═'*70}")
            
            true_labels = np.where(y_true[idx] == 1)[0]
            pred_labels = np.where(y_pred[idx] == 1)[0]
            
            # Correct predictions (True Positives)
            correct = set(true_labels) & set(pred_labels)
            # Missed labels (False Negatives)
            missed = set(true_labels) - set(pred_labels)
            # Extra labels (False Positives)
            extra = set(pred_labels) - set(true_labels)
            
            print(f"\nGROUND TRUTH (y)")
            print(f"  Total: {len(true_labels)} labels")
            if len(true_labels) > 0:
                for i, label_idx in enumerate(true_labels, 1):
                    label_name = target_names[label_idx] if target_names is not None else f"Label_{label_idx}"
                    print(f"  {i}. {label_name}")
            else:
                print(f"  (No labels)")
            
            print(f"\nPREDICTION (y')")
            print(f"  Total: {len(pred_labels)} labels")
            if len(pred_labels) > 0:
                for i, label_idx in enumerate(pred_labels, 1):
                    label_name = target_names[label_idx] if target_names is not None else f"Label_{label_idx}"
                    status = "✓" if label_idx in correct else "✗"
                    print(f"  {i}. {status} {label_name}")
            else:
                print(f"  (No labels)")
            
            print(f"\nANALYSIS")
            print(f"   ✓ Correct:       {len(correct)} labels")
            if correct:
                for label_idx in correct:
                    label_name = target_names[label_idx] if target_names is not None else f"Label_{label_idx}"
                    print(f"      • {label_name}")
            
            print(f"   ✗ Missed:        {len(missed)} labels (False Negatives)")
            if missed:
                for label_idx in missed:
                    label_name = target_names[label_idx] if target_names is not None else f"Label_{label_idx}"
                    print(f"      • {label_name}")
            
            print(f"   ⚠ Extra:         {len(extra)} labels (False Positives)")
            if extra:
                for label_idx in extra:
                    label_name = target_names[label_idx] if target_names is not None else f"Label_{label_idx}"
                    print(f"      • {label_name}")
            
            # Performance metrics for this sample
            precision = len(correct) / len(pred_labels) if len(pred_labels) > 0 else 0
            recall = len(correct) / len(true_labels) if len(true_labels) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n   SAMPLE METRICS:")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
        
        print("\n" + "="*70)
    
    def compare_predictions_summary(self, y_true, y_pred, target_names=None):
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        n_samples = y_true.shape[0]
        n_labels = y_true.shape[1]
        
        print("\n" + "="*70)
        print("PREDICTION COMPARISON SUMMARY")
        print("="*70)
        
        # Perfect matches
        perfect_matches = 0
        partial_matches = 0
        no_matches = 0
        
        for i in range(n_samples):
            if np.array_equal(y_true[i], y_pred[i]):
                perfect_matches += 1
            elif np.any((y_true[i] == 1) & (y_pred[i] == 1)):
                partial_matches += 1
            else:
                no_matches += 1
        
        print(f"\nSample-Level Results (out of {n_samples:,} samples):")
        print(f"  ✓ Perfect matches:  {perfect_matches:6,} ({perfect_matches/n_samples*100:5.2f}%)")
        print(f"  ⚠ Partial matches:  {partial_matches:6,} ({partial_matches/n_samples*100:5.2f}%)")
        print(f"  ✗ No matches:       {no_matches:6,} ({no_matches/n_samples*100:5.2f}%)")
        
        # Label-level statistics
        print(f"\nLabel-Level Statistics (Top 10):")
        
        for i in range(min(10, n_labels)):
            true_count = np.sum(y_true[:, i])
            pred_count = np.sum(y_pred[:, i])
            correct_count = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            
            label_name = target_names[i] if target_names is not None else f"Label_{i}"
            
            precision = correct_count / pred_count if pred_count > 0 else 0
            recall = correct_count / true_count if true_count > 0 else 0
            
            print(f"\n  Label: {label_name}")
            print(f"    Ground Truth: {int(true_count):4} | Predicted: {int(pred_count):4} | Correct: {int(correct_count):4}")
            print(f"    Precision: {precision:.3f} | Recall: {recall:.3f}")
        
        print("\n" + "="*70)
    
    def plot_all_analyses(self, y_train, y_train_pred, y_val, y_val_pred, 
                         y_test, y_test_pred, train_metrics, val_metrics, 
                         test_metrics, target_names=None):
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*70)
        
        # Convert all to NumPy
        y_train = self._to_numpy(y_train)
        y_train_pred = self._to_numpy(y_train_pred)
        y_val = self._to_numpy(y_val)
        y_val_pred = self._to_numpy(y_val_pred)
        y_test = self._to_numpy(y_test)
        y_test_pred = self._to_numpy(y_test_pred)
        
        # Generate plots (simplified for HPC)
        try:
            self.plot_metrics_comparison(train_metrics, val_metrics, test_metrics)
            print("✓ Saved metrics comparison")
        except Exception as e:
            print(f"⚠ Could not generate metrics comparison: {e}")
        
        print("\n✓ Visualizations complete!")
    
    def plot_metrics_comparison(self, train_metrics, val_metrics, test_metrics=None):
        metrics_to_plot = ['f1_micro', 'f1_macro', 'precision_micro', 
                          'recall_micro', 'subset_accuracy', 'jaccard_score']
        metric_names = ['F1 (Micro)', 'F1 (Macro)', 'Precision (Micro)', 
                       'Recall (Micro)', 'Subset Accuracy', 'Jaccard Score']
        
        datasets = ['Train', 'Validation']
        all_metrics = [train_metrics, val_metrics]
        
        if test_metrics:
            datasets.append('Test')
            all_metrics.append(test_metrics)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for idx, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            values = [m[metric_key] for m in all_metrics]
            
            bars = axes[idx].bar(datasets, values, color=colors[:len(datasets)], 
                               alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[idx].set_ylabel('Score', fontsize=12, fontweight='bold')
            axes[idx].set_title(metric_name, fontsize=14, fontweight='bold')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                             f'{height:.3f}',
                             ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, '01_metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()