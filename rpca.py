# rpca.py
# RPCA using SCIPY SVDS — no sklearn needed

import numpy as np
from scipy.sparse.linalg import svds


class RobustPCA:
    """
    Fast RPCA using Scipy SVD.
    Decomposes M into L (background) + S (foreground).
    """

    def __init__(self, lam=None, mu=None, tol=1e-5, max_iter=100, n_components=10):
        self.lam          = lam
        self.mu           = mu
        self.tol          = tol
        self.max_iter     = max_iter
        self.n_components = n_components

    def _shrink(self, X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    def _svd_threshold(self, X, tau):
        k = min(self.n_components, min(X.shape) - 1)
        try:
            U, sigma, Vt = svds(X, k=k)
            # svds returns ascending order — reverse to descending
            idx   = np.argsort(sigma)[::-1]
            U     = U[:, idx]
            sigma = sigma[idx]
            Vt    = Vt[idx, :]
        except Exception:
            # Fallback to full SVD
            U, sigma, Vt = np.linalg.svd(X, full_matrices=False)

        sigma_thresh = np.maximum(sigma - tau, 0)
        return U @ np.diag(sigma_thresh) @ Vt

    def fit(self, M, progress_callback=None):
        rows, cols = M.shape
        print(f"\n🔧 RPCA started — Matrix: {M.shape}")
        print(f"   Using Scipy SVD (top {self.n_components} components)")

        if self.lam is None:
            self.lam = 1.0 / np.sqrt(max(rows, cols))

        if self.mu is None:
            self.mu = rows * cols / (4.0 * np.sum(np.abs(M)))

        print(f"   λ={self.lam:.5f}   μ={self.mu:.5f}")
        print(f"   Tolerance={self.tol}   Max Iter={self.max_iter}\n")

        S      = np.zeros_like(M)
        Y      = np.zeros_like(M)
        L      = np.zeros_like(M)
        mu_inv = 1.0 / self.mu
        norm_M = np.linalg.norm(M, 'fro')

        print(f"{'Iter':>5}  {'Error':>10}  {'Status'}")
        print("-" * 35)

        for i in range(self.max_iter):

            L        = self._svd_threshold(M - S + mu_inv * Y, mu_inv)
            S        = self._shrink(M - L + mu_inv * Y, self.lam * mu_inv)
            residual = M - L - S
            Y        = Y + self.mu * residual
            error    = np.linalg.norm(residual, 'fro') / norm_M

            status = "running..."
            if error < self.tol:
                status = "✅ CONVERGED"

            if (i + 1) % 5 == 0 or i == 0 or error < self.tol:
                print(f"{i+1:>5}  {error:>10.6f}  {status}")

            if progress_callback:
                progress_callback(i + 1, error)

            if error < self.tol:
                print(f"\n✅ Done in {i+1} iterations!")
                break

        else:
            print(f"\n⚠️ Stopped at max iterations ({self.max_iter})")
            print(f"   Final error: {error:.6f}")

        self.L = L
        self.S = S
        return L, S