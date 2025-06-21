import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for iteration in range(self.max_iter):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = np.mean(predictions - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if iteration > 0 and np.mean(np.abs(dw)) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
                
        print(f"Training completed in {iteration + 1} iterations")
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)