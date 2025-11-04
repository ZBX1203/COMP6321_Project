
from sklearn.datasets import fetch_rcv1
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import json
import os
from datetime import datetime
from scipy.sparse import issparse


class RCV1DataLoader:

    def __init__(self, data_dir='data'):
        """
        Args:
            data_dir: Directory to store data files
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)

    def download_and_split(self, test_size=0.2, val_size=0.1, random_state=42,
                          convert_to_dense=False, sample_size=None,full=False):
        """
        Args:
            test_size: Proportion for complete test set
            val_size: Proportion for small validation/test set
            random_state: Random seed for reproducibility
            convert_to_dense: If True, convert sparse to dense (requires lots of memory)
            sample_size: If provided, randomly sample this many documents (e.g., 50000)
        """
        print("="*70)
        print("DOWNLOADING RCV1-v2 DATASET")
        print("="*70)

        # Download dataset
        rcv1 = fetch_rcv1(subset='all', download_if_missing=True)

        X = rcv1.data
        y = rcv1.target
        target_names = rcv1.target_names

        print(f"\nTotal samples: {X.shape[0]:,}")
        print(f"Features: {X.shape[1]:,}")
        print(f"Categories: {y.shape[1]}")
        print(f"Data format: {'Sparse' if issparse(X) else 'Dense'}")

        if sample_size is not None and sample_size < X.shape[0]:
            if full:
                sample_size = X.shape[0]
            print(f"\nSampling {sample_size:,} documents for faster training...")
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Sampled data: {X.shape[0]:,} samples")

        if convert_to_dense:
            print("\nConverting to dense format (this may take time and memory)...")
            if issparse(X):
                X = X.toarray()
            if issparse(y):
                y = y.toarray()
            print("✓ Converted to dense NumPy arrays")
        else:
            if issparse(y):
                print("\nConverting labels to dense format...")
                y = y.toarray()
            print("✓ Features kept as sparse, labels converted to dense")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )

        print("\n" + "="*70)
        print("DATASET SPLIT")
        print("="*70)
        total = X.shape[0]
        print(f"Training set:   {X_train.shape[0]:,} samples ({X_train.shape[0]/total*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/total*100:.1f}%)")
        print(f"Test set:       {X_test.shape[0]:,} samples ({X_test.shape[0]/total*100:.1f}%)")

        y_train = np.asarray(y_train)
        y_val = np.asarray(y_val)
        y_test = np.asarray(y_test)

        self._save_data(X_train, y_train, X_val, y_val, X_test, y_test, target_names)

        return X_train, y_train, X_val, y_val, X_test, y_test, target_names

    def _save_data(self, X_train, y_train, X_val, y_val, X_test, y_test, target_names):
        print("\n" + "="*70)
        print("SAVING DATA")
        print("="*70)

        with open(os.path.join(self.data_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(X_train, f)
        with open(os.path.join(self.data_dir, 'train_labels.pkl'), 'wb') as f:
            pickle.dump(y_train, f)
        print(f"✓ Saved training set")

        with open(os.path.join(self.data_dir, 'val_data.pkl'), 'wb') as f:
            pickle.dump(X_val, f)
        with open(os.path.join(self.data_dir, 'val_labels.pkl'), 'wb') as f:
            pickle.dump(y_val, f)
        print(f"✓ Saved validation set")

        with open(os.path.join(self.data_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(X_test, f)
        with open(os.path.join(self.data_dir, 'test_labels.pkl'), 'wb') as f:
            pickle.dump(y_test, f)
        print(f"✓ Saved test set")

        labels_per_doc = np.sum(y_train, axis=1)
        metadata = {
            'save_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_train': int(X_train.shape[0]),
            'n_val': int(X_val.shape[0]),
            'n_test': int(X_test.shape[0]),
            'n_features': int(X_train.shape[1]),
            'n_categories': int(y_train.shape[1]),
            'target_names': target_names.tolist(),
            'avg_labels_per_doc': float(np.mean(labels_per_doc)),
            'is_sparse': issparse(X_train)
        }

        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata")

    def load_data(self, split='train'):
        """
        Args:
            split: 'train', 'val', or 'test'

        Returns:
            X, y: Features and labels (as NumPy arrays)
        """
        print(f"Loading {split} data...")

        with open(os.path.join(self.data_dir, f'{split}_data.pkl'), 'rb') as f:
            X = pickle.load(f)
        with open(os.path.join(self.data_dir, f'{split}_labels.pkl'), 'rb') as f:
            y = pickle.load(f)

        # Ensure y is NumPy array
        if issparse(y):
            y = y.toarray()
        y = np.asarray(y)

        print(f"Loaded {split} set: {X.shape[0]:,} samples")
        print(f"  Features: {'Sparse' if issparse(X) else 'Dense NumPy array'}")
        print(f"  Labels: NumPy array {y.shape}")
        return X, y

    def load_metadata(self):
        with open(os.path.join(self.data_dir, 'metadata.json'), 'r') as f:
            return json.load(f)