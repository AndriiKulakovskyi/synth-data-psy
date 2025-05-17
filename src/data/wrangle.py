import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Tuple, Any


class DataWrangler:
    """
    End-to-end dataframe pre-processor.

    Parameters
    ----------
    conversion_threshold : float, default 0.80
        Minimum share of non-null values that must convert to numeric
        before an object column is treated as numeric.
    cardinality_threshold : int, default 10
        Numeric columns with < threshold distinct values are treated
        as categoricals (avoids scaling one-hot-like/ordinal features).
    scale_numeric : bool, default True
        If True, apply `StandardScaler` to numeric columns.
    """

    def __init__(
        self,
        conversion_threshold: float = 0.80,
        cardinality_threshold: int = 10,
        scale_numeric: bool = True,
    ) -> None:
        self.conversion_threshold = conversion_threshold
        self.cardinality_threshold = cardinality_threshold
        self.scale_numeric = scale_numeric
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.nan_medians: Dict[str, float] = {}

    def _detect_and_cast(self, s: pd.Series) -> Tuple[pd.Series, bool]:
        s_clean = s.replace([np.inf, -np.inf], np.nan)
        if s_clean.dtype == "object":
            s_clean = s_clean.str.strip()

        coerced = pd.to_numeric(s_clean, errors="coerce")
        non_null, numeric_ok = s_clean.notna().sum(), coerced.notna().sum()

        if non_null and numeric_ok / non_null >= self.conversion_threshold:
            coerced = coerced.replace([np.inf, -np.inf], np.nan)
            return coerced, True
        return s_clean, False

    def wrangle(
        self, df: pd.DataFrame, *, fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Transform *df* in-place-safe manner and return
        (X_ready, params_dict) for downstream ML pipelines.

        Parameters
        ----------
        df   : input dataframe (left intact)
        fit  : if False, assumes encoders/scalers are already fitted;
               useful for transforming hold-out / test sets.

        Notes
        -----
        * Numeric NaNs ⇒ median per column.
        * Categorical NaNs ⇒ literal string ``"Missing"`` (is encoded).
        * Encoders & scalers stored so ``inverse_transform`` is trivial.
        """
        X = df.copy(deep=True)

        for col in X.columns:
            series, is_numeric = self._detect_and_cast(X[col])

            # (1) treat as *categorical* (either truly object or low-cardinality numeric)
            if (not is_numeric) or (
                is_numeric and series.nunique(dropna=True) < self.cardinality_threshold
            ):
                # fill NaNs before encoding
                series_filled = series.fillna("Missing").astype(str)

                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(series_filled)
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    X[col] = le.transform(series_filled)

            # (2) treat as *numeric*
            else:
                series = series.replace([np.inf, -np.inf], np.nan)

                # median from finite numbers only
                if fit:
                    finite = series.dropna()
                    median = finite.median() if not finite.empty else 0.0
                    self.nan_medians[col] = median
                else:
                    median = self.nan_medians[col]

                series_filled = series.fillna(median).astype(float)

                if self.scale_numeric:
                    if fit:
                        scaler = StandardScaler()
                        # never passes NaN/Inf now
                        X[col] = scaler.fit_transform(series_filled.values.reshape(-1, 1)).ravel()
                        self.scalers[col] = scaler
                    else:
                        X[col] = self.scalers[col].transform(series_filled.values.reshape(-1, 1)).ravel()
                else:
                    X[col] = series_filled

        params: Dict[str, Any] = {
            "encoders": self.encoders,
            "scalers": self.scalers,
            "nan_medians": self.nan_medians,
            "scale_numeric": self.scale_numeric,
        }
        return X, params

    def inverse_transform(
        self, X: pd.DataFrame, drop_scaled: bool = False
    ) -> pd.DataFrame:
        """
        Partially (or fully) reverse transformation.  Useful for:

        * inspection of feature importances on original scale
        * debugging / model explainability

        Parameters
        ----------
        drop_scaled : if True, scaled numerics are restored to original values;
                      otherwise, they stay standardised.
        """
        rev = X.copy(deep=True)

        # ▸ categorical
        for col, le in self.encoders.items():
            rev[col] = le.inverse_transform(rev[col].astype(int))

        # ▸ numeric
        if drop_scaled and self.scale_numeric:
            for col, scaler in self.scalers.items():
                col_arr = rev[col].values.reshape(-1, 1)
                rev[col] = scaler.inverse_transform(col_arr).ravel()

        # restore NaNs where appropriate
        for col, median in self.nan_medians.items():
            if pd.isna(median):
                continue
            rev.loc[rev[col] == median, col] = np.nan

        return rev


def wrangle(df: pd.DataFrame,
            conversion_threshold: float = 0.80,
            cardinality_threshold: int = 10,
            scale_numeric: bool = True,
            ) -> Tuple[pd.DataFrame, Dict[str, Any], DataWrangler]:
    """
    Stateless convenience wrapper; returns **(X_ready, params, fitted_wrangler)**.

    You can later reuse ``fitted_wrangler.wrangle(new_df, fit=False)`` on
    validation / test sets and call ``fitted_wrangler.inverse_transform``.

    >>> X_train_ready, params, wr = wrangle(train_df)
    >>> X_test_ready, _     = wr.wrangle(test_df, fit=False)
    """
    wr = DataWrangler(
        conversion_threshold=conversion_threshold,
        cardinality_threshold=cardinality_threshold,
        scale_numeric=scale_numeric,
    )
    X_ready, params = wr.wrangle(df, fit=True)
    return X_ready, params, wr
