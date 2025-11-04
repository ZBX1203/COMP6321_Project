
import numpy as np
from model import DNNClassifier
from data_loader import RCV1DataLoader
import sys


def test_specific_samples(sample_indices, model_path='models/dnn_model.pth'):
    """
    Args:
        sample_indices: List of sample indices to test
        model_path: Path to saved model
    """
    
    print("="*70)
    print("TESTING SPECIFIC SAMPLES")
    print("="*70)
    
    print("\nLoading model...")
    model = DNNClassifier.load(model_path)
    
    print("Loading test data...")
    data_loader = RCV1DataLoader(data_dir='data')
    X_test, y_test = data_loader.load_data('test')
    metadata = data_loader.load_metadata()
    target_names = metadata['target_names']
    
    if hasattr(y_test, 'toarray'):
        y_test = y_test.toarray()
    y_test = np.asarray(y_test)
    
    for idx in sample_indices:
        if idx >= X_test.shape[0]:
            print(f"\n⚠ Sample {idx} out of range (max: {X_test.shape[0]-1})")
            continue
        
        print("\n" + "="*70)
        print(f"SAMPLE INDEX: {idx}")
        print("="*70)
        
        X_sample = X_test[idx:idx+1]
        y_true = y_test[idx]
        
        y_pred = model.predict(X_sample)[0]
        
        true_indices = np.where(y_true == 1)[0]
        pred_indices = np.where(y_pred == 1)[0]
        
        true_labels = [target_names[i] for i in true_indices]
        pred_labels = [target_names[i] for i in pred_indices]
        
        correct = set(true_indices) & set(pred_indices)
        missed = set(true_indices) - set(pred_indices)
        extra = set(pred_indices) - set(true_indices)
        
        print(f"\nGROUND TRUTH ({len(true_labels)} labels):")
        for label in true_labels:
            print(f"  • {label}")
        
        print(f"\nPREDICTED ({len(pred_labels)} labels):")
        for i in pred_indices:
            label = target_names[i]
            status = "✓" if i in correct else "✗"
            print(f"  {status} {label}")
        
        if missed:
            print(f"\nMISSED ({len(missed)} labels):")
            for i in missed:
                print(f"  • {target_names[i]}")
        
        if extra:
            print(f"\nEXTRA ({len(extra)} labels):")
            for i in extra:
                print(f"  • {target_names[i]}")
        
        precision = len(correct) / len(pred_indices) if len(pred_indices) > 0 else 0
        recall = len(correct) / len(true_indices) if len(true_indices) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMETRICS:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # From command line: python test_specific_samples.py 0 10 50 100
        sample_indices = [int(x) for x in sys.argv[1:]]
    else:
        # Default samples
        sample_indices = [0, 10, 50, 100, 500, 1000]
    
    test_specific_samples(sample_indices)