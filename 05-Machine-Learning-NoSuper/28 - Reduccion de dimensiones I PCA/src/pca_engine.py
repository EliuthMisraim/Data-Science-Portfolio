"""Motor modular de estandarización + PCA (capa de ingeniería).

Esta capa no contiene visualización ni lógica exploratoria: solo cálculo
reutilizable por el notebook (``notebooks/``) y la app (``app/``).
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Un módulo de librería solo obtiene su logger; la configuración
# (logging.basicConfig) le corresponde al notebook o a la app.
logger = logging.getLogger(__name__)


class PCAProcessor:
    """Estandariza (z-score) y aplica PCA; separa ajuste (fit) de inferencia (transform)."""

    def __init__(self, n_components: Optional[int] = 2) -> None:
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.feature_columns_: Optional[List[str]] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.components_dataframe_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Ajuste e inferencia
    # ------------------------------------------------------------------ #
    def fit_transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Ajusta scaler y PCA con ``df`` y devuelve las proyecciones."""
        self._check_columns(df, feature_columns)
        logger.info("Iniciando transformación PCA con %s componentes.", self.n_components)

        X_scaled = self.scaler.fit_transform(df[feature_columns].to_numpy(dtype=float))
        X_pca = self.pca.fit_transform(X_scaled)

        self.feature_columns_ = list(feature_columns)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.components_dataframe_ = pd.DataFrame(
            self.pca.components_,
            columns=self.feature_columns_,
            index=[f"PC{i + 1}" for i in range(self.pca.n_components_)],
        )

        logger.info(
            "Varianza explicada acumulada: %.2f%%",
            float(np.sum(self.explained_variance_ratio_)) * 100,
        )
        return X_pca

    def transform(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """Proyecta nuevas observaciones con el modelo ya ajustado (sin reajustar)."""
        self._check_fitted()
        if list(feature_columns) != self.feature_columns_:
            raise ValueError(
                "Las columnas deben coincidir (y en el mismo orden) con las del ajuste: "
                f"{self.feature_columns_}"
            )
        self._check_columns(df, feature_columns)
        X_scaled = self.scaler.transform(df[feature_columns].to_numpy(dtype=float))
        return self.pca.transform(X_scaled)

    # ------------------------------------------------------------------ #
    # Resultados para interpretación
    # ------------------------------------------------------------------ #
    def get_factor_map(self) -> pd.DataFrame:
        """Eigenvectores (norma 1): dirección de cada PC en el espacio de variables."""
        self._check_fitted()
        return self.components_dataframe_

    def get_loadings(self) -> pd.DataFrame:
        """Cargas factoriales = correlación variable-componente (círculo de correlación).

        loading = eigenvector * sqrt(eigenvalor). Con variables estandarizadas, su
        valor está en [-1, 1] y la longitud de la flecha mide qué tan bien el plano
        representa a la variable.
        """
        self._check_fitted()
        loadings = self.pca.components_ * np.sqrt(self.pca.explained_variance_)[:, None]
        return pd.DataFrame(
            loadings, columns=self.feature_columns_, index=self.components_dataframe_.index
        )

    # ------------------------------------------------------------------ #
    # Validaciones internas
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.feature_columns_ is None:
            raise RuntimeError("PCAProcessor no está ajustado: llama primero a fit_transform().")

    @staticmethod
    def _check_columns(df: pd.DataFrame, feature_columns: List[str]) -> None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Columnas ausentes en el DataFrame: {missing}")
