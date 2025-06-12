import re
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.api.types import infer_dtype
from typing import Any, Dict, List, Sequence
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


class DataTransformer:
    """
    A class to transform data for machine learning by handling categorical encoding 
    and numerical scaling with inverse transformation capabilities.
    """
    
    def __init__(self, categorical_threshold: int = 10, save_transforms: bool = True, save_folder: str = 'DATA/processed'):
        """
        Initialize the DataTransformer.
        
        Parameters:
        -----------
        categorical_threshold : int
            Maximum number of unique values for a column to be considered categorical (default: 10)
        save_transforms : bool
            Whether to save transformation dictionaries to files (default: True)
        save_folder : str
            Folder path where to save transformation files (default: 'DATA/processed')
        """
        self.categorical_threshold = categorical_threshold
        self.save_transforms = save_transforms
        self.save_folder = save_folder
        self.label_encoders = {}
        self.scalers = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.transform_info = {}
        self.is_fitted = False
        
    def _detect_column_types(self, df: pd.DataFrame) -> None:
        """Detect which columns are categorical and which are numerical."""
        self.categorical_columns = []
        self.numerical_columns = []
        
        for column in df.columns:
            # Skip if all values are NaN
            if df[column].isna().all():
                continue
                
            # Check if column is already numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                unique_count = df[column].nunique()
                
                # If numeric column has few unique values, treat as categorical
                if unique_count <= self.categorical_threshold:
                    self.categorical_columns.append(column)
                    print(f"Column '{column}' detected as categorical (numeric with {unique_count} unique values)")
                else:
                    self.numerical_columns.append(column)
                    print(f"Column '{column}' detected as numerical")
            else:
                # Non-numeric columns are categorical
                unique_count = df[column].nunique()
                self.categorical_columns.append(column)
                print(f"Column '{column}' detected as categorical (non-numeric with {unique_count} unique values)")
    
    def fit(self, df: pd.DataFrame) -> 'DataTransformer':
        """
        Fit the transformer on the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to fit transformers on
            
        Returns:
        --------
        DataTransformer
            Self for method chaining
        """
        print("Fitting MLDataTransformer...")
        
        # Detect column types
        self._detect_column_types(df)
        
        # Fit label encoders for categorical columns
        for column in self.categorical_columns:
            le = LabelEncoder()
            # Handle NaN values by filling them temporarily
            non_null_data = df[column].dropna()
            if len(non_null_data) > 0:
                le.fit(non_null_data)
                self.label_encoders[column] = le
                
                # Store mapping for inverse transformation
                self.transform_info[column] = {
                    'type': 'categorical',
                    'classes': le.classes_.tolist(),
                    'mapping': {str(k): int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
                }
                print(f"Fitted LabelEncoder for '{column}' with {len(le.classes_)} classes")
        
        # Fit scalers for numerical columns
        for column in self.numerical_columns:
            scaler = StandardScaler()
            # Handle NaN values by using only non-null data for fitting
            non_null_data = df[[column]].dropna()
            if len(non_null_data) > 0:
                scaler.fit(non_null_data)
                self.scalers[column] = scaler
                
                # Store scaling parameters for inverse transformation
                self.transform_info[column] = {
                    'type': 'numerical',
                    'mean': float(scaler.mean_[0]),
                    'std': float(scaler.scale_[0])
                }
                print(f"Fitted StandardScaler for '{column}' (mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f})")
        
        self.is_fitted = True
        
        # Save transformation info if requested
        if self.save_transforms:
            self._save_transform_info()
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted transformers.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transforming data. Call fit() first.")
        
        df_transformed = df.copy()
        
        # Transform categorical columns
        for column in self.categorical_columns:
            if column in df_transformed.columns and column in self.label_encoders:
                le = self.label_encoders[column]
                
                # Handle unseen categories and NaN values
                def safe_transform(x):
                    if pd.isna(x):
                        return np.nan
                    if x in le.classes_:
                        return le.transform([x])[0]
                    else:
                        print(f"Warning: Unseen category '{x}' in column '{column}', assigning -1")
                        return -1  # Assign -1 for unseen categories
                
                df_transformed[column] = df[column].apply(safe_transform)
                print(f"Transformed categorical column '{column}'")
        
        # Transform numerical columns
        for column in self.numerical_columns:
            if column in df_transformed.columns and column in self.scalers:
                scaler = self.scalers[column]
                
                # Handle NaN values
                mask = df_transformed[column].notna()
                if mask.any():
                    # Transform and ensure float dtype to avoid casting issues
                    scaled_values = scaler.transform(
                        df_transformed.loc[mask, [column]]
                    ).flatten()
                    
                    # Convert column to float64 to avoid casting issues
                    if df_transformed[column].dtype in ['Int64', 'int64']:
                        df_transformed[column] = df_transformed[column].astype('float64')
                    
                    df_transformed.loc[mask, column] = scaled_values
                print(f"Transformed numerical column '{column}'")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the transformer and transform the data in one step.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Transformed dataframe
        """
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data back to original scale.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transformed dataframe to inverse transform
            
        Returns:
        --------
        pd.DataFrame
            Data in original scale
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse transforming.")
        
        df_inverse = df.copy()
        
        # Inverse transform categorical columns
        for column in self.categorical_columns:
            if column in df_inverse.columns and column in self.label_encoders:
                le = self.label_encoders[column]
                
                def safe_inverse_transform(x):
                    if pd.isna(x) or x == -1:
                        return np.nan
                    try:
                        x_int = int(x)
                        if 0 <= x_int < len(le.classes_):
                            return le.inverse_transform([x_int])[0]
                        else:
                            return np.nan
                    except (ValueError, IndexError):
                        return np.nan
                
                df_inverse[column] = df[column].apply(safe_inverse_transform)
                print(f"Inverse transformed categorical column '{column}'")
        
        # Inverse transform numerical columns
        for column in self.numerical_columns:
            if column in df_inverse.columns and column in self.scalers:
                scaler = self.scalers[column]
                
                # Handle NaN values
                mask = df_inverse[column].notna()
                if mask.any():
                    df_inverse.loc[mask, column] = scaler.inverse_transform(
                        df_inverse.loc[mask, [column]]
                    ).flatten()
                print(f"Inverse transformed numerical column '{column}'")
        
        return df_inverse
    
    def _save_transform_info(self):
        """Save transformation information and fitted objects to files."""
        try:
            # Create directory if it doesn't exist
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)
            
            # Save transformation info as JSON
            transform_info_path = Path(self.save_folder) / 'transform_info.json'
            with open(transform_info_path, 'w') as f:
                json.dump(self.transform_info, f, indent=2)
            
            # Save the entire transformer object using pickle
            transformer_path = Path(self.save_folder) / 'ml_transformer.pkl'
            with open(transformer_path, 'wb') as f:
                pickle.dump(self, f)
            
            print(f"Transformation info saved to '{transform_info_path}'")
            print(f"Complete transformer object saved to '{transformer_path}'")
            
        except Exception as e:
            print(f"Warning: Could not save transformation info: {e}")
    
    def load_transformer(self, filepath: str = None) -> 'DataTransformer':
        """
        Load a previously saved transformer.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the saved transformer file. If None, uses self.save_folder/ml_transformer.pkl
            
        Returns:
        --------
        DataTransformer
            Loaded transformer
        """
        if filepath is None:
            filepath = Path(self.save_folder) / 'ml_transformer.pkl'
        
        try:
            with open(filepath, 'rb') as f:
                loaded_transformer = pickle.load(f)
            print(f"Transformer loaded from '{filepath}'")
            return loaded_transformer
        except Exception as e:
            print(f"Error loading transformer: {e}")
            return None
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the features and their transformations.
        
        Returns:
        --------
        Dict
            Dictionary containing feature transformation information
        """
        return {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'total_features': len(self.categorical_columns) + len(self.numerical_columns),
            'transform_info': self.transform_info
        }


def load_data(path: str, drop_columns: List[str]) -> pd.DataFrame:
    try:
        raw_data = pd.read_csv(path, sep=';', low_memory=False)
        raw_data = raw_data.drop(columns=drop_columns)
        return raw_data
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def drop_cols_by_missing_count(df: pd.DataFrame, threshold: int, inplace: bool = False) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame; got {type(df).__name__}")
    if threshold < 0 or threshold > 100:
        raise ValueError("`threshold` must be between 0 and 100")

    # Normalize threshold to fraction
    frac_threshold = threshold if threshold <= 1 else threshold / 100.0

    # Compute fraction of missing per column
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > frac_threshold].index

    if inplace:
        df.drop(columns=cols_to_drop, inplace=True)
        return df
    else:
        return df.drop(columns=cols_to_drop)

def drop_rows_by_missing_count(df: pd.DataFrame, threshold: int, inplace: bool = False) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame; got {type(df).__name__}")
    if threshold < 0 or threshold > 100:
        raise ValueError("`threshold` must be between 0 and 100")

    # Normalize threshold to fraction
    frac_threshold = threshold if threshold <= 1 else threshold / 100.0

    # Compute fraction of missing per row
    missing_frac = df.isna().mean(axis=1)
    rows_to_drop = missing_frac[missing_frac > frac_threshold].index

    if inplace:
        df.drop(index=rows_to_drop, inplace=True)
        return df
    else:
        return df.drop(index=rows_to_drop)

def get_unique_column_values(df: pd.DataFrame) -> dict:
    unique_data = {}

    for col in df.columns:
        series = df[col].dropna()
        unique_values = series.unique()
        # Convert numpy arrays to lists for JSON serialization
        if hasattr(unique_values, 'tolist'):
            unique_data[col] = unique_values.tolist()
        else:
            unique_data[col] = list(unique_values)

    # Use a custom JSON encoder to handle other numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    return unique_data

def clean_and_convert_value(value):
        """Clean and convert a single value to numerical format."""
        if pd.isna(value):
            return np.nan
        
        # If already a number, return as is
        if isinstance(value, (int, float)):
            return value
        
        # Convert to string and clean
        str_value = str(value).strip()
        
        # Handle empty strings
        if not str_value:
            return np.nan
        
        # Handle range values like "50 - 75", "50-75", "50 to 75"
        range_patterns = [
            r'(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)',  # 50-75, 50 - 75
            r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',      # 50 to 75
            r'(\d+(?:\.\d+)?)\s*[àa]\s*(\d+(?:\.\d+)?)'     # 50 à 75 (French)
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, str_value, re.IGNORECASE)
            if match:
                try:
                    val1, val2 = float(match.group(1)), float(match.group(2))
                    return (val1 + val2) / 2
                except ValueError:
                    continue
        
        # Handle comparison operators like "<1", ">5", "<=10", ">=20"
        comparison_match = re.search(r'([<>=]+)\s*(\d+(?:\.\d+)?)', str_value)
        if comparison_match:
            try:
                return float(comparison_match.group(2))
            except ValueError:
                pass
        
        # Handle "approximately" or "about" values like "~50", "≈50", "about 50"
        approx_patterns = [
            r'[~≈]\s*(\d+(?:\.\d+)?)',
            r'(?:about|environ|approximately|approx\.?)\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in approx_patterns:
            match = re.search(pattern, str_value, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Extract first number from strings like "34 ans", "25 years", "50kg"
        number_in_text = re.search(r'(\d+(?:\.\d+)?)', str_value)
        if number_in_text:
            try:
                return float(number_in_text.group(1))
            except ValueError:
                pass
        
        # If no patterns match, try direct conversion
        try:
            # Remove common non-numeric characters and try conversion
            cleaned = re.sub(r'[^\d.-]', '', str_value)
            if cleaned:
                return float(cleaned)
        except ValueError:
            pass
        
        return None  # Could not convert

def convert_to_numerical(df: pd.DataFrame, min_conversion_rate: float = 0.6) -> pd.DataFrame:
    """
    Convert columns to numerical values where possible, handling various string formats.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    min_conversion_rate : float
        Minimum fraction of values that must be convertible for column conversion (default: 0.6)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted numerical columns
    """
    df_converted = df.copy()
    
    for column in df.columns:
        # Skip if column is already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            continue
        
        # Test conversion on non-null values
        non_null_values = df[column].dropna()
        if len(non_null_values) == 0:
            continue
        
        # Try to convert each value
        converted_values = []
        successful_conversions = 0
        
        for value in non_null_values:
            converted = clean_and_convert_value(value)
            if converted is not None:
                successful_conversions += 1
                converted_values.append(converted)
            else:
                converted_values.append(value)
        
        # Check if enough values were successfully converted
        conversion_rate = successful_conversions / len(non_null_values)
        
        if conversion_rate >= min_conversion_rate:
            # Apply conversion to the entire column
            new_column = df[column].apply(clean_and_convert_value)
            
            # Determine if the result should be int or float
            numeric_values = new_column.dropna()
            if len(numeric_values) > 0:
                # Check if all numeric values are whole numbers
                if all(isinstance(val, (int, float)) and val == int(val) for val in numeric_values if not pd.isna(val)):
                    new_column = new_column.astype('Int64')  # Nullable integer type
                else:
                    new_column = pd.to_numeric(new_column, errors='coerce')
                
                df_converted[column] = new_column
                print(f"Converted column '{column}' to numerical (conversion rate: {conversion_rate:.2%})")
        else:
            print(f"Skipped column '{column}' - low conversion rate: {conversion_rate:.2%}")
    
    return df_converted

def split_ml_data(df_ml_ready: pd.DataFrame, ml_transformer: DataTransformer = None) -> tuple:
    """
    Split ML-ready dataframe into numerical and categorical components.
    
    Parameters:
    -----------
    df_ml_ready : pd.DataFrame
        ML-ready dataframe with encoded categorical and scaled numerical columns
    ml_transformer : DataTransformer, optional
        The fitted transformer object to get column type information
    
    Returns:
    --------
    tuple
        (numerical_df, categorical_info_df) where:
        - numerical_df: DataFrame with only numerical columns
        - categorical_info_df: DataFrame with categorical columns and their category counts
    """
    
    if ml_transformer is not None:
        # Use transformer information if available
        numerical_columns = ml_transformer.numerical_columns
        categorical_columns = ml_transformer.categorical_columns
    else:
        # Fallback: detect based on data characteristics
        numerical_columns = []
        categorical_columns = []
        
        for column in df_ml_ready.columns:
            # Skip columns with all NaN values
            if df_ml_ready[column].isna().all():
                continue
            
            # Check unique values and data type
            unique_count = df_ml_ready[column].nunique()
            
            # Heuristic: if column has few unique values and they're integers, likely categorical
            if unique_count <= 20 and df_ml_ready[column].dtype in ['int64', 'Int64', 'float64']:
                # Check if values look like encoded categories (0, 1, 2, etc.)
                non_null_values = df_ml_ready[column].dropna()
                if len(non_null_values) > 0:
                    min_val = non_null_values.min()
                    max_val = non_null_values.max()
                    # If values are in range [0, unique_count-1] and mostly integers, likely categorical
                    if min_val >= 0 and max_val < unique_count * 1.5:
                        categorical_columns.append(column)
                    else:
                        numerical_columns.append(column)
                else:
                    numerical_columns.append(column)
            else:
                numerical_columns.append(column)
    
    # Create numerical dataframe
    numerical_df = df_ml_ready[numerical_columns].copy()
    
    # Create categorical dataframe with category information
    categorical_data = []
    for column in categorical_columns:
        if column in df_ml_ready.columns:
            unique_count = df_ml_ready[column].nunique()
            unique_values = sorted(df_ml_ready[column].dropna().unique())
            
            # Get original category names if transformer is available
            original_categories = None
            if ml_transformer is not None and column in ml_transformer.transform_info:
                if ml_transformer.transform_info[column]['type'] == 'categorical':
                    original_categories = ml_transformer.transform_info[column]['classes']
            
            categorical_data.append({
                'column_name': column,
                'num_categories': unique_count,
                'encoded_values': unique_values,
                'original_categories': original_categories if original_categories else 'Not available'
            })
    
    # Create categorical info dataframe
    if categorical_data:
        categorical_info_df = pd.DataFrame(categorical_data)
    else:
        categorical_info_df = pd.DataFrame(columns=['column_name', 'num_categories', 'encoded_values', 'original_categories'])
    
    # Also create a dataframe with just the categorical columns and their data
    categorical_df = df_ml_ready[categorical_columns].copy() if categorical_columns else pd.DataFrame()
    
    print(f"Data split summary:")
    print(f"- Numerical columns: {len(numerical_columns)}")
    print(f"- Categorical columns: {len(categorical_columns)}")
    print(f"- Total columns: {len(numerical_columns) + len(categorical_columns)}")
    
    return numerical_df, categorical_df, categorical_info_df

def analyze_categorical_distribution(categorical_df: pd.DataFrame, categorical_info_df: pd.DataFrame, 
                                   save_folder: str = None) -> pd.DataFrame:
    """
    Analyze the distribution of categorical variables.
    
    Parameters:
    -----------
    categorical_df : pd.DataFrame
        DataFrame with categorical columns
    categorical_info_df : pd.DataFrame
        DataFrame with categorical column information
    save_folder : str, optional
        Folder to save the analysis results
    
    Returns:
    --------
    pd.DataFrame
        Distribution analysis for each categorical column
    """
    
    distribution_analysis = []
    
    for _, row in categorical_info_df.iterrows():
        column_name = row['column_name']
        
        if column_name in categorical_df.columns:
            # Get value counts
            value_counts = categorical_df[column_name].value_counts().sort_index()
            total_count = categorical_df[column_name].count()
            
            # Calculate percentages
            percentages = (value_counts / total_count * 100).round(2)
            
            # Create distribution info
            distribution_info = {
                'column_name': column_name,
                'total_non_null': total_count,
                'num_categories': row['num_categories'],
                'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_percentage': percentages.iloc[0] if len(percentages) > 0 else 0,
                'distribution': dict(zip(value_counts.index, value_counts.values)),
                'percentage_distribution': dict(zip(percentages.index, percentages.values))
            }
            
            distribution_analysis.append(distribution_info)
    
    distribution_df = pd.DataFrame(distribution_analysis)
    
    # Save analysis if folder is provided
    if save_folder and not distribution_df.empty:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        distribution_df.to_csv(f'{save_folder}/categorical_distribution_analysis.csv', sep=';', index=False)
        print(f"Categorical distribution analysis saved to '{save_folder}/categorical_distribution_analysis.csv'")
    
    return distribution_df

def impute_missing_data(numerical_df: pd.DataFrame, categorical_df: pd.DataFrame, 
                       numerical_strategy: str = 'median', categorical_strategy: str = 'most_frequent',
                       knn_neighbors: int = 5, save_folder: str = None) -> tuple:
    """
    Impute missing values in numerical and categorical dataframes.
    
    Parameters:
    -----------
    numerical_df : pd.DataFrame
        DataFrame with numerical columns containing missing values
    categorical_df : pd.DataFrame  
        DataFrame with categorical columns containing missing values
    numerical_strategy : str
        Strategy for numerical imputation: 'mean', 'median', 'most_frequent', 'constant', 'knn'
    categorical_strategy : str
        Strategy for categorical imputation: 'most_frequent', 'constant', 'forward_fill', 'backward_fill'
    knn_neighbors : int
        Number of neighbors for KNN imputation (only used if numerical_strategy='knn')
    save_folder : str, optional
        Folder to save imputation parameters
    
    Returns:
    --------
    tuple
        (numerical_imputed, categorical_imputed, imputation_info) where:
        - numerical_imputed: DataFrame with imputed numerical values
        - categorical_imputed: DataFrame with imputed categorical values  
        - imputation_info: Dictionary containing imputation parameters for reproducibility
    """
    
    imputation_info = {
        'numerical_strategy': numerical_strategy,
        'categorical_strategy': categorical_strategy,
        'numerical_imputers': {},
        'categorical_imputers': {},
        'missing_counts_before': {},
        'missing_counts_after': {}
    }
    
    # Record missing counts before imputation
    if not numerical_df.empty:
        imputation_info['missing_counts_before']['numerical'] = numerical_df.isnull().sum().to_dict()
    if not categorical_df.empty:
        imputation_info['missing_counts_before']['categorical'] = categorical_df.isnull().sum().to_dict()
    
    print("Starting missing data imputation...")
    print(f"Numerical strategy: {numerical_strategy}")
    print(f"Categorical strategy: {categorical_strategy}")
    
    # Impute numerical data
    numerical_imputed = numerical_df.copy()
    if not numerical_df.empty and numerical_df.isnull().any().any():
        print(f"\nImputing numerical data ({numerical_df.shape[1]} columns)...")
        
        if numerical_strategy == 'knn':
            # Use KNN imputation
            knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
            numerical_imputed.iloc[:, :] = knn_imputer.fit_transform(numerical_df)
            imputation_info['numerical_imputers']['knn'] = {
                'n_neighbors': knn_neighbors,
                'feature_names': numerical_df.columns.tolist()
            }
            print(f"Applied KNN imputation with {knn_neighbors} neighbors")
            
        else:
            # Use SimpleImputer for other strategies
            if numerical_strategy == 'constant':
                fill_value = 0  # Can be customized
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                imputation_info['numerical_imputers']['constant_value'] = fill_value
            else:
                imputer = SimpleImputer(strategy=numerical_strategy)
            
            numerical_imputed.iloc[:, :] = imputer.fit_transform(numerical_df)
            
            # Store imputation values for each column
            if hasattr(imputer, 'statistics_'):
                imputation_info['numerical_imputers']['values'] = dict(zip(
                    numerical_df.columns, imputer.statistics_
                ))
            
            print(f"Applied {numerical_strategy} imputation to {numerical_df.shape[1]} columns")
    
    # Impute categorical data
    categorical_imputed = categorical_df.copy()
    if not categorical_df.empty and categorical_df.isnull().any().any():
        print(f"\nImputing categorical data ({categorical_df.shape[1]} columns)...")
        
        if categorical_strategy == 'most_frequent':
            # Use mode imputation
            imputer = SimpleImputer(strategy='most_frequent')
            categorical_imputed.iloc[:, :] = imputer.fit_transform(categorical_df)
            
            # Store the most frequent values
            imputation_info['categorical_imputers']['most_frequent_values'] = dict(zip(
                categorical_df.columns, imputer.statistics_
            ))
            print(f"Applied most frequent imputation to {categorical_df.shape[1]} columns")
            
        elif categorical_strategy == 'constant':
            # Use constant value (e.g., -1 for unknown category)
            fill_value = -1
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
            categorical_imputed.iloc[:, :] = imputer.fit_transform(categorical_df)
            imputation_info['categorical_imputers']['constant_value'] = fill_value
            print(f"Applied constant value ({fill_value}) imputation")
            
        elif categorical_strategy == 'forward_fill':
            # Forward fill
            categorical_imputed = categorical_df.fillna(method='ffill')
            print("Applied forward fill imputation")
            
        elif categorical_strategy == 'backward_fill':
            # Backward fill
            categorical_imputed = categorical_df.fillna(method='bfill')
            print("Applied backward fill imputation")
        
        # For any remaining NaN values after forward/backward fill, use most frequent
        if categorical_strategy in ['forward_fill', 'backward_fill'] and categorical_imputed.isnull().any().any():
            print("Filling remaining NaN values with most frequent values...")
            for column in categorical_imputed.columns:
                if categorical_imputed[column].isnull().any():
                    mode_value = categorical_imputed[column].mode()
                    if len(mode_value) > 0:
                        categorical_imputed[column].fillna(mode_value[0], inplace=True)
                    else:
                        categorical_imputed[column].fillna(-1, inplace=True)  # Fallback
    
    # Record missing counts after imputation
    if not numerical_imputed.empty:
        imputation_info['missing_counts_after']['numerical'] = numerical_imputed.isnull().sum().to_dict()
    if not categorical_imputed.empty:
        imputation_info['missing_counts_after']['categorical'] = categorical_imputed.isnull().sum().to_dict()
    
    # Print summary
    print("\nImputation completed!")
    if not numerical_df.empty:
        missing_before_num = numerical_df.isnull().sum().sum()
        missing_after_num = numerical_imputed.isnull().sum().sum()
        print(f"Numerical data: {missing_before_num} → {missing_after_num} missing values")
    
    if not categorical_df.empty:
        missing_before_cat = categorical_df.isnull().sum().sum()
        missing_after_cat = categorical_imputed.isnull().sum().sum()
        print(f"Categorical data: {missing_before_cat} → {missing_after_cat} missing values")
    
    # Save imputation info
    if save_folder:
        try:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            with open(f'{save_folder}/imputation_info.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_safe_info = {}
                for key, value in imputation_info.items():
                    if isinstance(value, dict):
                        json_safe_info[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                             for k, v in value.items()}
                    else:
                        json_safe_info[key] = value
                
                json.dump(json_safe_info, f, indent=2)
            print(f"Imputation info saved to '{save_folder}/imputation_info.json'")
        except Exception as e:
            print(f"Warning: Could not save imputation info: {e}")
    
    return numerical_imputed, categorical_imputed, imputation_info

def analyze_missing_data(numerical_df: pd.DataFrame, categorical_df: pd.DataFrame, 
                        save_folder: str = None) -> pd.DataFrame:
    """
    Analyze missing data patterns in numerical and categorical dataframes.
    
    Parameters:
    -----------
    numerical_df : pd.DataFrame
        DataFrame with numerical columns
    categorical_df : pd.DataFrame
        DataFrame with categorical columns
    save_folder : str, optional
        Folder to save the analysis
    
    Returns:
    --------
    pd.DataFrame
        Missing data analysis summary
    """
    
    missing_analysis = []
    
    # Analyze numerical columns
    if not numerical_df.empty:
        for column in numerical_df.columns:
            missing_count = numerical_df[column].isnull().sum()
            total_count = len(numerical_df)
            missing_percentage = (missing_count / total_count) * 100
            
            missing_analysis.append({
                'column_name': column,
                'data_type': 'numerical',
                'total_rows': total_count,
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'non_missing_count': total_count - missing_count
            })
    
    # Analyze categorical columns  
    if not categorical_df.empty:
        for column in categorical_df.columns:
            missing_count = categorical_df[column].isnull().sum()
            total_count = len(categorical_df)
            missing_percentage = (missing_count / total_count) * 100
            
            missing_analysis.append({
                'column_name': column,
                'data_type': 'categorical',
                'total_rows': total_count,
                'missing_count': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'non_missing_count': total_count - missing_count
            })
    
    missing_df = pd.DataFrame(missing_analysis)
    
    if not missing_df.empty:
        # Sort by missing percentage descending
        missing_df = missing_df.sort_values('missing_percentage', ascending=False)
        
        # Save analysis
        if save_folder:
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            missing_df.to_csv(f'{save_folder}/missing_data_analysis.csv', sep=';', index=False)
            print(f"Missing data analysis saved to '{save_folder}/missing_data_analysis.csv'")
        
        # Print summary
        print("\nMissing Data Analysis Summary:")
        print(f"Total columns analyzed: {len(missing_df)}")
        columns_with_missing = missing_df[missing_df['missing_count'] > 0]
        print(f"Columns with missing data: {len(columns_with_missing)}")
        
        if len(columns_with_missing) > 0:
            print(f"Highest missing percentage: {columns_with_missing['missing_percentage'].max():.2f}%")
            print(f"Average missing percentage: {columns_with_missing['missing_percentage'].mean():.2f}%")
            
            print("\nTop 10 columns with most missing data:")
            top_missing = columns_with_missing.head(10)[['column_name', 'data_type', 'missing_percentage']]
            print(top_missing.to_string(index=False))
    
    return missing_df


    