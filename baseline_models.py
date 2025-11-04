
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import issparse
import pickle
import time


class BaselineModel:
    
    def __init__(self, model_type='logistic', name=None, **kwargs):
        """
        Args:
            model_type: Type of model ('logistic', 'sgd', 'rf', 'nb', 'svm', 'dt')
            name: Custom name for the model
            **kwargs: Additional parameters for the model
        """
        self.model_type = model_type
        self.name = name or model_type.upper()
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
        self.training_time = 0
        
    def build_model(self):
        
        if self.model_type == 'logistic':
            # Logistic Regression - use Binary Relevance approach
            from sklearn.multiclass import OneVsRestClassifier
            base_model = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    solver='lbfgs',
                    n_jobs=1
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
            self.model = base_model
            
        elif self.model_type == 'sgd':
            # SGD Classifier - use Binary Relevance
            from sklearn.multiclass import OneVsRestClassifier
            base_model = OneVsRestClassifier(
                SGDClassifier(
                    loss='log_loss',
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    n_jobs=1
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
            self.model = base_model
            
        elif self.model_type == 'rf':
            # Random Forest
            base_model = RandomForestClassifier(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 20),
                random_state=self.kwargs.get('random_state', 42),
                n_jobs=1,
                verbose=0
            )
            n_jobs = self.kwargs.get('n_jobs', -1)
            self.model = MultiOutputClassifier(base_model, n_jobs=n_jobs)
            
        elif self.model_type == 'nb':
            # Naive Bayes
            base_model = MultinomialNB(
                alpha=self.kwargs.get('alpha', 1.0)
            )
            n_jobs = self.kwargs.get('n_jobs', -1)
            self.model = MultiOutputClassifier(base_model, n_jobs=n_jobs)
            
        elif self.model_type == 'svm':
            # Linear SVM
            from sklearn.multiclass import OneVsRestClassifier
            base_model = OneVsRestClassifier(
                LinearSVC(
                    max_iter=self.kwargs.get('max_iter', 1000),
                    random_state=self.kwargs.get('random_state', 42),
                    dual=False
                ),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
            self.model = base_model
            
        elif self.model_type == 'dt':
            # Decision Tree
            base_model = DecisionTreeClassifier(
                max_depth=self.kwargs.get('max_depth', 20),
                random_state=self.kwargs.get('random_state', 42)
            )
            n_jobs = self.kwargs.get('n_jobs', -1)
            self.model = MultiOutputClassifier(base_model, n_jobs=n_jobs)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def fit(self, X, y):
        print(f"\nTraining {self.name}...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Convert labels to dense if needed
        if issparse(y):
            print("  Converting labels to dense...")
            y = y.toarray()
        
        # For Naive Bayes, need to convert sparse to dense
        if self.model_type == 'nb' and issparse(X):
            print("  Converting features to dense for Naive Bayes...")
            X = X.toarray()
        
        # Train
        start_time = time.time()
        self.model.fit(X, y)
        self.training_time = time.time() - start_time
        
        self.is_fitted = True
        print(f"✓ {self.name} training completed in {self.training_time:.2f}s")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # For Naive Bayes, convert to dense
        if self.model_type == 'nb' and issparse(X):
            X = X.toarray()
        
        start_time = time.time()
        predictions = self.model.predict(X)
        prediction_time = time.time() - start_time
        
        return predictions, prediction_time
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ {self.name} saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


    
    def __init__(self, strategy='frequent', name=None):
        """
        Args:
            strategy: 'frequent' (most frequent labels) or 'random'
            name: Custom name
        """
        self.strategy = strategy
        self.name = name or f"Dummy-{strategy}"
        self.is_fitted = False
        self.label_frequencies = None
        self.n_labels = None
        self.training_time = 0
    
    def fit(self, X, y):
        """Fit the dummy model"""
        print(f"\nTraining {self.name}...")
        
        start_time = time.time()
        
        if issparse(y):
            y = y.toarray()
        
        self.n_labels = y.shape[1]
        
        if self.strategy == 'frequent':
            # Learn most frequent labels
            self.label_frequencies = np.mean(y, axis=0)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"✓ {self.name} training completed in {self.training_time:.4f}s")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        n_samples = X.shape[0]
        
        if self.strategy == 'frequent':
            # Predict most frequent labels (above 0.5 frequency)
            predictions = np.tile(
                (self.label_frequencies > 0.5).astype(int),
                (n_samples, 1)
            )
        else:  # random
            # Random predictions with same frequency as training
            predictions = (np.random.rand(n_samples, self.n_labels) > 0.5).astype(int)
        
        prediction_time = time.time() - start_time
        
        return predictions, prediction_time
    
    def save(self, filepath):
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ {self.name} saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model