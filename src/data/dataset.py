from __future__ import annotations
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Tuple, Literal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    QuantileTransformer,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ScalerStrategy = Literal["standard", "minmax", "quantile", None]
CatEncoding    = Literal["onehot", "ordinal", None]


def _make_scaler(strategy: ScalerStrategy):
    if strategy == "standard":
        return StandardScaler()
    if strategy == "minmax":
        return MinMaxScaler()
    if strategy == "quantile":
        return QuantileTransformer(
            n_quantiles=1000, output_distribution="normal", random_state=0
        )
    if strategy is None:
        return None
    raise ValueError(f"Unknown scaling strategy: {strategy!r}")


def _make_encoder(encoding: CatEncoding):
    if encoding == "onehot":
        return OneHotEncoder(
            sparse=False, handle_unknown="ignore", dtype=np.float32
        )
    if encoding == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    if encoding is None:
        return None
    raise ValueError(f"Unknown cat_encoding: {encoding!r}")


@dataclass
class PreprocessedData:
    X_train: np.ndarray
    X_test:  np.ndarray
    y_train: Optional[np.ndarray]
    y_test:  Optional[np.ndarray]
    scaler:  Optional[Any]
    encoder: Optional[Any]
    mapping: Dict[str, Any]
    train_dataset:  TabularDataset
    test_dataset:   TabularDataset


class TabularDataset(Dataset):
    def __init__(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """
        PyTorch Dataset wrapping numeric + categorical features,
        with optional labels y.
        """
        if X_num.shape[0] != X_cat.shape[0]:
            raise ValueError("X_num and X_cat must have the same number of rows")
        if y is not None:
            y_arr = np.asarray(y)
            if y_arr.shape[0] != X_num.shape[0]:
                raise ValueError("y must have the same length as X_num / X_cat")
            self.y = torch.from_numpy(y_arr)
        else:
            self.y = None

        self.X_num = torch.from_numpy(X_num).float()
        # categorical indices or one-hot floats
        self.X_cat = (
            torch.from_numpy(X_cat).long()
            if np.issubdtype(X_cat.dtype, np.integer)
            else torch.from_numpy(X_cat).float()
        )

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X_num[idx], self.X_cat[idx], self.y[idx]
        return self.X_num[idx], self.X_cat[idx]

    def __len__(self) -> int:
        return self.X_num.shape[0]


def split_numerical_categorical(df: pd.DataFrame,
                                cardinality_threshold: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Split a DataFrame into numeric & categorical matrices.

    Args:
        df: Input DataFrame.
        cardinality_threshold: 
            If a numeric column has <= this many distinct values,
            it will be treated as categorical.

    Returns:
        numerical_matrix: shape (n_rows, n_numeric_cols)
        categorical_matrix: shape (n_rows, n_categorical_cols)
        mapping: {
            "numerical": { new_col_idx: original_col_name, … },
            "categorical": { new_col_idx: original_col_name, … },
            "original_order": [col1, col2, …],
            "index": df.index
        }

    Raises:
        TypeError, ValueError on bad inputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame")
    if not isinstance(cardinality_threshold, int) or cardinality_threshold < 0:
        raise ValueError("`cardinality_threshold` must be a non-negative int")

    numerical_cols: List[str]   = []
    categorical_cols: List[str] = []

    # decide for each column
    for col in df.columns:
        ser = df[col]
        if pd.api.types.is_numeric_dtype(ser):
            if ser.nunique(dropna=True) <= cardinality_threshold:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    # build matrices
    numerical_matrix   = df[numerical_cols].to_numpy()
    categorical_matrix = df[categorical_cols].to_numpy()

    # build mapping new_idx → name
    num_map = {i: name for i, name in enumerate(numerical_cols)}
    cat_map = {i: name for i, name in enumerate(categorical_cols)}

    mapping: Dict[str, Any] = {
        "numerical":    num_map,
        "categorical":  cat_map,
        "original_order": list(df.columns),
        "index":        df.index
    }

    return numerical_matrix, categorical_matrix, mapping


def reconstruct_dataframe(
    numerical_matrix: np.ndarray,
    categorical_matrix: np.ndarray,
    mapping: Dict[str, Any],
    dtype_map: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Rebuild the original DataFrame from two NumPy matrices + mapping.

    Args:
        numerical_matrix:    as returned by split_numerical_categorical
        categorical_matrix:  likewise
        mapping:             the dict returned by split_numerical_categorical
        dtype_map:           optional {col_name: dtype} to cast columns back

    Returns:
        DataFrame identical in shape, column order, and index to the original.
    """
    # basic checks
    if not isinstance(mapping, dict):
        raise TypeError("mapping must be the dict returned by split_numerical_categorical")
    orig_order = mapping.get("original_order")
    orig_index = mapping.get("index")
    num_map    = mapping.get("numerical")
    cat_map    = mapping.get("categorical")

    # rebuild
    df = pd.DataFrame(index=orig_index)

    # place numericals
    for new_idx, name in num_map.items():
        df[name] = numerical_matrix[:, new_idx]

    # place categoricals
    for new_idx, name in cat_map.items():
        df[name] = categorical_matrix[:, new_idx]

    # re‐order to original
    df = df[orig_order]

    # recast dtypes if requested
    if dtype_map:
        for col, dt in dtype_map.items():
            df[col] = df[col].astype(dt)

    return df


def preprocess_data(
    *,
    num_mat: np.ndarray,
    cat_mat: np.ndarray,
    mapping: Dict[str, Any],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    test_size: float                  = 0.2,
    random_state: int | None          = 42,
    transform: bool                   = True,
    scaling_strategy: ScalerStrategy  = "standard",
    cat_encoding:   CatEncoding       = "onehot",
    num_imputer_strategy: str         = "median",
    cat_imputer_strategy: str         = "most_frequent",
    stratify: bool                    = True,
) -> PreprocessedData:
    """
    Splits, (optionally) transforms, and wraps into PyTorch Datasets.
    Supports y=None for unsupervised use.

    Returns:
        PreprocessedData with raw & transformed arrays, fitted scaler/encoder (or None),
        the original mapping, and PyTorch train/test datasets.
    """
    # ─── basic checks ──────────────────────────────────────────────────────────
    if not isinstance(num_mat, np.ndarray) or not isinstance(cat_mat, np.ndarray):
        raise TypeError("num_mat and cat_mat must be NumPy arrays")
    if num_mat.shape[0] != cat_mat.shape[0]:
        raise ValueError("num_mat and cat_mat must have same n_rows")
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.shape[0] != num_mat.shape[0]:
            raise ValueError("y length must match num_mat rows")
    else:
        y_arr = None

    # ─── train/test split ─────────────────────────────────────────────────────
    stratify_arg = None
    if y_arr is not None and stratify and np.unique(y_arr).size > 1:
        stratify_arg = y_arr

    if y_arr is not None:
        split = train_test_split(
            num_mat, cat_mat, y_arr,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
        num_train, num_test, cat_train, cat_test, y_train, y_test = split
    else:
        split = train_test_split(
            num_mat, cat_mat,
            test_size=test_size,
            random_state=random_state,
        )
        num_train, num_test, cat_train, cat_test = split
        y_train = y_test = None

    # ─── fast path: raw split without transforms ────────────────────────────────
    if not transform:
        X_train = np.hstack([num_train, cat_train])
        X_test  = np.hstack([num_test,  cat_test])
        train_ds = TabularDataset(num_train, cat_train, y_train)
        test_ds  = TabularDataset(num_test,  cat_test,  y_test)
        return PreprocessedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=None,
            encoder=None,
            mapping=mapping,
            train_dataset=train_ds,
            test_dataset=test_ds,
        )

    # ─── build & fit transformation pipelines ─────────────────────────────────
    # numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy=num_imputer_strategy))]
    scaler = _make_scaler(scaling_strategy)
    if scaler is not None:
        num_steps.append(("scaler", scaler))
    num_pipe = Pipeline(num_steps)

    # categorical pipeline
    cat_steps = [("imputer", SimpleImputer(strategy=cat_imputer_strategy))]
    encoder = _make_encoder(cat_encoding)
    if encoder is not None:
        cat_steps.append(("encoder", encoder))
    cat_pipe = Pipeline(cat_steps)

    # fit/transform
    num_train_p = num_pipe.fit_transform(num_train)
    num_test_p  = num_pipe.transform(num_test)
    cat_train_p = cat_pipe.fit_transform(cat_train)
    cat_test_p  = cat_pipe.transform(cat_test)

    # ─── combine & wrap into PyTorch datasets ─────────────────────────────────
    X_train = np.hstack([num_train_p, cat_train_p])
    X_test  = np.hstack([num_test_p,  cat_test_p])
    train_ds = TabularDataset(num_train_p, cat_train_p, y_train)
    test_ds  = TabularDataset(num_test_p,  cat_test_p,  y_test)

    return PreprocessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        encoder=encoder,
        mapping=mapping,
        train_dataset=train_ds,
        test_dataset=test_ds,
    )


def update_ema(target_params, source_params, rate=0.999):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)
