import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.common import loader_to_numpy


class PCABaseline:
    def __init__(self, variance_threshold=0.90, n_components=None):
        """
        variance_threshold : used when n_components is None to find optimal_n
        n_components       : if set, skips the full PCA fit and uses this fixed dim
        """
        self.variance_threshold = variance_threshold
        self.n_components = n_components
        self.pca_full = PCA()
        self.pca_fixed = PCA()
        self.optimal_n = n_components
        self.cumulative_variance = []

    def fit(self, train_loader, test_loader=None):
        """
        If n_components is fixed: only fits a PCA at that dimension.
        If n_components is None: fits full PCA to find optimal_n from variance threshold.
        """
        X_train = loader_to_numpy(loader=train_loader)
        if test_loader is not None:
            X_test = loader_to_numpy(loader=test_loader)
            X_all = np.concatenate([X_train, X_test], axis=0)
        else:
            X_all = X_train

        if self.n_components is not None:
            # Fixed mode — only fit at specified dimension
            self.pca_fixed = PCA(n_components=self.n_components).fit(X_all)
        else:
            # Full mode — fit all components to find optimal_n
            self.pca_full = PCA().fit(X_all)
            self.cumulative_variance = np.cumsum(self.pca_full.explained_variance_ratio_)
            self.optimal_n = np.where(self.cumulative_variance >= self.variance_threshold)[0][0] + 1
            # Also fit a fixed PCA at optimal_n for reconstruction use
            self.pca_fixed = PCA(n_components=self.optimal_n).fit(X_all)

        return self

    def transform(self, loader, return_names=False):
        """Project data into PCA space. Optionally return audio names for lyrics combination."""
        self._check_fitted()
        X, names = [], []

        for batch in loader:
            x, paths, _ = batch  # unpack 3-tuple, ignore labels
            X.append(x.numpy().reshape(len(x), -1))
            names.extend(paths)

        X = np.concatenate(X, axis=0)
        latents = self.pca_fixed.transform(X)

        if return_names:
            return latents, names
        return latents

    def reconstruct(self, loader):
        """Transform then inverse transform for reconstruction comparison with VAE."""
        self._check_fitted()
        X = loader_to_numpy(loader=loader)
        
        return self.pca_fixed.inverse_transform(self.pca_fixed.transform(X))

    def reconstruction_error(self, loader):
        """Compute per-sample MSE reconstruction error, comparable to VAE recon loss."""
        X = loader_to_numpy(loader=loader)
        X_reconstructed = self.pca_fixed.inverse_transform(self.pca_fixed.transform(X))
        
        return np.mean((X - X_reconstructed) ** 2)

    def plot(self, figsize=(10, 6)):
        """Plot cumulative explained variance curve. Only available in full mode."""
        if self.pca_full is None:
            raise RuntimeError("plot() is only available when n_components is not fixed. Run in full mode.")

        plt.figure(figsize=figsize)
        plt.plot(self.cumulative_variance, marker='o', linestyle='-')
        plt.axhline(y=self.variance_threshold, color='r', linestyle='--', label=f'{int(self.variance_threshold * 100)}% Variance Threshold')
        plt.axvline(x=self.optimal_n, color='g', linestyle='--', label=f'Optimal n={self.optimal_n}')
        plt.title('PCA Explained Variance vs. Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Optimal components for {int(self.variance_threshold * 100)}% variance: {self.optimal_n}")

    def summary(self):
        """Print a summary of the fitted PCA."""
        self._check_fitted()
        n = self.n_components or self.optimal_n
        variance = (
            self.cumulative_variance[n - 1]
            if self.cumulative_variance is not None
            else np.sum(self.pca_fixed.explained_variance_ratio_)
        )
        mode = "fixed" if self.n_components is not None else "auto"
        print(f"Mode               : {mode}")
        print(f"Variance threshold : {int(self.variance_threshold * 100)}%")
        print(f"Components used    : {n}")
        print(f"Variance explained : {variance:.4f}")

    def _check_fitted(self):
        if self.pca_fixed is None:
            raise RuntimeError("PCABaseline is not fitted yet. Call .fit() first.")