
import numpy as np
import torch
import pickle
import json
from datetime import datetime
from model import DNNClassifier
from data_loader import RCV1DataLoader
from evaluator import MultiLabelEvaluator
import pandas as pd
import os


class ModelTester:
    
    def __init__(self, model_path='models/dnn_model.pth', output_dir='test_results'):
        """
        Args:
            model_path: Path to saved model
            output_dir: Directory to save test results
        """
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*70)
        print("MODEL TESTER INITIALIZED")
        print("="*70)
        print(f"Model path: {model_path}")
        print(f"Output directory: {output_dir}")
    
    def load_model(self):
        print("\nLoading trained model...")
        self.model = DNNClassifier.load(self.model_path)
        print("✓ Model loaded successfully")
    
    def load_test_data(self):
        print("\nLoading test data...")
        data_loader = RCV1DataLoader(data_dir='data')
        
        self.X_test, self.y_test = data_loader.load_data('test')
        metadata = data_loader.load_metadata()
        self.target_names = metadata['target_names']
        
        print(f"✓ Loaded {self.X_test.shape[0]:,} test samples")
        print(f"  Features: {self.X_test.shape[1]:,}")
        print(f"  Labels: {len(self.target_names)}")
    
    def test_and_save_batch(self, batch_size=100, num_batches=5):
        """
        Args:
            batch_size: Number of samples per batch
            num_batches: Number of batches to process
        """
        print("\n" + "="*70)
        print("TESTING MODEL ON BATCHES")
        print("="*70)
        
        total_samples = min(batch_size * num_batches, self.X_test.shape[0])
        
        all_results = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.X_test.shape[0])
            
            if start_idx >= self.X_test.shape[0]:
                break
            
            print(f"\nProcessing Batch {batch_idx + 1}/{num_batches}")
            print(f"Samples: {start_idx} to {end_idx-1}")
            
            X_batch = self.X_test[start_idx:end_idx]
            y_batch = self.y_test[start_idx:end_idx]
            
            y_pred_batch = self.model.predict(X_batch)
            
            batch_results = self._process_batch(
                X_batch, y_batch, y_pred_batch, 
                start_idx, end_idx, batch_idx
            )
            
            all_results.extend(batch_results)
        
            self._save_batch_results(batch_results, batch_idx)
        
        self._save_all_results(all_results)
        
        print("\n" + "="*70)
        print("✓ TESTING COMPLETE")
        print("="*70)
        print(f"Total samples tested: {len(all_results)}")
        print(f"Results saved to: {self.output_dir}/")
    
    def _process_batch(self, X_batch, y_batch, y_pred_batch, start_idx, end_idx, batch_idx):
        batch_results = []
        
        if hasattr(y_batch, 'toarray'):
            y_batch = y_batch.toarray()
        y_batch = np.asarray(y_batch)
        y_pred_batch = np.asarray(y_pred_batch)
        
        for i in range(len(y_batch)):
            sample_idx = start_idx + i
        
            true_label_indices = np.where(y_batch[i] == 1)[0]
            true_labels = [self.target_names[idx] for idx in true_label_indices]
            
            pred_label_indices = np.where(y_pred_batch[i] == 1)[0]
            pred_labels = [self.target_names[idx] for idx in pred_label_indices]
            
            correct = set(true_label_indices) & set(pred_label_indices)
            missed = set(true_label_indices) - set(pred_label_indices)
            extra = set(pred_label_indices) - set(true_label_indices)
            
            precision = len(correct) / len(pred_label_indices) if len(pred_label_indices) > 0 else 0
            recall = len(correct) / len(true_label_indices) if len(true_label_indices) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if len(true_label_indices) == len(pred_label_indices) == len(correct):
                status = "PERFECT"
            elif len(correct) > 0:
                status = "PARTIAL"
            else:
                status = "NO_MATCH"
            
            result = {
                'sample_index': sample_idx,
                'batch_index': batch_idx,
                'num_true_labels': len(true_labels),
                'num_pred_labels': len(pred_labels),
                'true_labels': true_labels,
                'predicted_labels': pred_labels,
                'correct_labels': [self.target_names[idx] for idx in correct],
                'missed_labels': [self.target_names[idx] for idx in missed],
                'extra_labels': [self.target_names[idx] for idx in extra],
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'status': status,
                'num_correct': len(correct),
                'num_missed': len(missed),
                'num_extra': len(extra)
            }
            
            batch_results.append(result)
        
        return batch_results
    
    def _save_batch_results(self, batch_results, batch_idx):
        """Save results for a single batch"""
        
        json_path = os.path.join(self.output_dir, f'batch_{batch_idx}_results.json')
        with open(json_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        txt_path = os.path.join(self.output_dir, f'batch_{batch_idx}_readable.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"BATCH {batch_idx} RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for result in batch_results:
                f.write("-"*70 + "\n")
                f.write(f"Sample Index: {result['sample_index']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write("-"*70 + "\n")
                
                f.write(f"\nGROUND TRUTH ({result['num_true_labels']} labels):\n")
                for label in result['true_labels']:
                    f.write(f"  • {label}\n")
                
                f.write(f"\nPREDICTED ({result['num_pred_labels']} labels):\n")
                for label in result['predicted_labels']:
                    status_symbol = "✓" if label in result['correct_labels'] else "✗"
                    f.write(f"  {status_symbol} {label}\n")
                
                if result['missed_labels']:
                    f.write(f"\nMISSED ({result['num_missed']} labels):\n")
                    for label in result['missed_labels']:
                        f.write(f"  • {label}\n")
                
                if result['extra_labels']:
                    f.write(f"\nEXTRA ({result['num_extra']} labels):\n")
                    for label in result['extra_labels']:
                        f.write(f"  • {label}\n")
                
                f.write(f"\nMETRICS:\n")
                f.write(f"  Precision: {result['precision']:.3f}\n")
                f.write(f"  Recall:    {result['recall']:.3f}\n")
                f.write(f"  F1-Score:  {result['f1_score']:.3f}\n")
                f.write("\n")
        
        print(f"  ✓ Saved batch {batch_idx} results")
    
    def _save_all_results(self, all_results):
        """Save combined results from all batches"""
        
        json_path = os.path.join(self.output_dir, 'all_results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        csv_path = os.path.join(self.output_dir, 'all_results.csv')
        df = pd.DataFrame([
            {
                'Sample_Index': r['sample_index'],
                'Batch_Index': r['batch_index'],
                'Num_True_Labels': r['num_true_labels'],
                'Num_Pred_Labels': r['num_pred_labels'],
                'Num_Correct': r['num_correct'],
                'Num_Missed': r['num_missed'],
                'Num_Extra': r['num_extra'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1_Score': r['f1_score'],
                'Status': r['status'],
                'True_Labels': '|'.join(r['true_labels']),
                'Predicted_Labels': '|'.join(r['predicted_labels'])
            }
            for r in all_results
        ])
        df.to_csv(csv_path, index=False)
        
        summary_path = os.path.join(self.output_dir, 'summary_statistics.txt')
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total samples tested: {len(all_results)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            statuses = [r['status'] for r in all_results]
            f.write("STATUS DISTRIBUTION:\n")
            f.write(f"  Perfect matches:  {statuses.count('PERFECT'):4d} ({statuses.count('PERFECT')/len(statuses)*100:5.2f}%)\n")
            f.write(f"  Partial matches:  {statuses.count('PARTIAL'):4d} ({statuses.count('PARTIAL')/len(statuses)*100:5.2f}%)\n")
            f.write(f"  No matches:       {statuses.count('NO_MATCH'):4d} ({statuses.count('NO_MATCH')/len(statuses)*100:5.2f}%)\n\n")
            
            avg_precision = np.mean([r['precision'] for r in all_results])
            avg_recall = np.mean([r['recall'] for r in all_results])
            avg_f1 = np.mean([r['f1_score'] for r in all_results])
            
            f.write("AVERAGE METRICS:\n")
            f.write(f"  Precision: {avg_precision:.4f}\n")
            f.write(f"  Recall:    {avg_recall:.4f}\n")
            f.write(f"  F1-Score:  {avg_f1:.4f}\n\n")
            
            avg_true = np.mean([r['num_true_labels'] for r in all_results])
            avg_pred = np.mean([r['num_pred_labels'] for r in all_results])
            avg_correct = np.mean([r['num_correct'] for r in all_results])
            
            f.write("LABEL STATISTICS:\n")
            f.write(f"  Avg true labels per sample:      {avg_true:.2f}\n")
            f.write(f"  Avg predicted labels per sample: {avg_pred:.2f}\n")
            f.write(f"  Avg correct labels per sample:   {avg_correct:.2f}\n")
        
        print(f"\n✓ Saved combined results:")
        print(f"  - {json_path}")
        print(f"  - {csv_path}")
        print(f"  - {summary_path}")
    
    def print_sample_results(self, num_samples=5):
        """Print results for a few samples"""
        
        json_path = os.path.join(self.output_dir, 'all_results.json')
        if not os.path.exists(json_path):
            print("No results found. Run test_and_save_batch() first.")
            return
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        print("\n" + "="*70)
        print("SAMPLE RESULTS")
        print("="*70)
        
        for i, result in enumerate(results[:num_samples]):
            print(f"\n{'═'*70}")
            print(f"SAMPLE {i+1} (Index: {result['sample_index']})")
            print(f"Status: {result['status']}")
            print(f"{'═'*70}")
            
            print(f"\nGROUND TRUTH ({result['num_true_labels']} labels):")
            for label in result['true_labels']:
                print(f"  • {label}")
            
            print(f"\nPREDICTED ({result['num_pred_labels']} labels):")
            for label in result['predicted_labels']:
                status = "✓" if label in result['correct_labels'] else "✗"
                print(f"  {status} {label}")
            
            print(f"\nMETRICS:")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall:    {result['recall']:.3f}")
            print(f"  F1-Score:  {result['f1_score']:.3f}")


def main():
    """Main testing function"""
    
    print("\n" + "="*70)
    print(" MODEL TESTING AND RESULT SAVING")
    print("="*70)
    
    tester = ModelTester(
        model_path='models/dnn_model.pth',
        output_dir='test_results'
    )
    
    tester.load_model()
    tester.load_test_data()
    
    tester.test_and_save_batch(
        batch_size=100,   # 100 samples per batch
        num_batches=10    # Test 10 batches (1000 samples total)
    )

    tester.print_sample_results(num_samples=5)
    
    print("\n" + "="*70)
    print(" TESTING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved in 'test_results/' directory:")
    print("  - batch_X_results.json: JSON format for each batch")
    print("  - batch_X_readable.txt: Human-readable format for each batch")
    print("  - all_results.json: All results in JSON")
    print("  - all_results.csv: All results in CSV (for Excel/analysis)")
    print("  - summary_statistics.txt: Overall statistics")
    print("\n")


if __name__ == '__main__':
    main()