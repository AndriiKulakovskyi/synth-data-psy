import json
import logging
import textwrap
import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
import re

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from typing import Any, Dict, List, Sequence

import ollama
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict


from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin


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

    with open('unique_data.json', 'w') as f:
        json.dump(unique_data, f, cls=NumpyEncoder, indent=2)
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

class MLDataTransformer:
    """
    A class to transform data for machine learning by handling categorical encoding 
    and numerical scaling with inverse transformation capabilities.
    """
    
    def __init__(self, categorical_threshold: int = 10, save_transforms: bool = True, save_folder: str = 'DATA/processed'):
        """
        Initialize the MLDataTransformer.
        
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
    
    def fit(self, df: pd.DataFrame) -> 'MLDataTransformer':
        """
        Fit the transformer on the data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe to fit transformers on
            
        Returns:
        --------
        MLDataTransformer
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
            scaler = MinMaxScaler()
            # Handle NaN values by using only non-null data for fitting
            non_null_data = df[[column]].dropna()
            if len(non_null_data) > 0:
                scaler.fit(non_null_data)
                self.scalers[column] = scaler
                
                # Store scaling parameters for inverse transformation
                self.transform_info[column] = {
                    'type': 'numerical',
                    'min': float(scaler.data_min_[0]),
                    'max': float(scaler.data_max_[0]),
                    'scale': float(scaler.scale_[0]),
                    'data_range': float(scaler.data_range_[0])
                }
                print(f"Fitted MinMaxScaler for '{column}' (range: {scaler.data_min_[0]:.2f} to {scaler.data_max_[0]:.2f})")
        
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
    
    def load_transformer(self, filepath: str = None) -> 'MLDataTransformer':
        """
        Load a previously saved transformer.
        
        Parameters:
        -----------
        filepath : str, optional
            Path to the saved transformer file. If None, uses self.save_folder/ml_transformer.pkl
            
        Returns:
        --------
        MLDataTransformer
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

if __name__ == "__main__":
    folder_path = 'FACE/processed'
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    raw_data = load_data('FACE/neuropsy_v0.csv', ['usubjid_neuropsychologie', 'visitnum_neuropsychologie', 'visit_neuropsychologie', 'visit_neuropsychologie'])
    print(raw_data.head())

    data_cleaned = drop_cols_by_missing_count(raw_data, threshold=25)
    data_cleaned = drop_rows_by_missing_count(data_cleaned, threshold=25)
    
    # Convert columns to numerical where possible
    print("\nConverting columns to numerical...")
    data_numerical = convert_to_numerical(data_cleaned, min_conversion_rate=0.6)
    data_numerical.to_csv(f'{folder_path}/data_numerical.csv', sep=';', index=False)
    
    print(f"\nDataframe shape after numerical conversion: {data_numerical.shape}")
    print(f"Data types after conversion:")
    print(data_numerical.dtypes.value_counts())
    
    # Initialize and fit the transformer
    ml_transformer = MLDataTransformer(categorical_threshold=10, save_transforms=True, save_folder=folder_path)
    data_ml_ready = ml_transformer.fit_transform(data_numerical)
    
    print(f"\nDataframe shape after ML transformations: {data_ml_ready.shape}")
    print("\nFeature transformation summary:")
    feature_info = ml_transformer.get_feature_info()
    print(f"- Categorical columns: {len(feature_info['categorical_columns'])}")
    print(f"- Numerical columns: {len(feature_info['numerical_columns'])}")
    print(f"- Total features: {feature_info['total_features']}")

    correlation_matrix = data_ml_ready.corr(method='pearson')

    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(20, 20))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()  # Ensure proper layout
    
    # Save the figure BEFORE showing it
    plt.savefig(f'{folder_path}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Correlation matrix heatmap saved to '{folder_path}/correlation_matrix.png'")
    
    # Show the plot after saving
    plt.show()
    
    # Apply ML transformations
    print("\n" + "="*50)
    print("APPLYING ML TRANSFORMATIONS")
    print("="*50)
    
    # Save the ML-ready dataset
    data_ml_ready.to_csv(f'{folder_path}/data_ml_ready.csv', sep=';', index=False)
    print(f"\nML-ready dataset saved to '{folder_path}/data_ml_ready.csv'")

    
    # Demonstrate inverse transformation
    print("\nDemonstrating inverse transformation...")
    sample_data = data_ml_ready.head(5)  # Take first 5 rows
    reconstructed_data = ml_transformer.inverse_transform(sample_data)
    
    print("Original sample (first 3 columns):")
    print(data_numerical.head(5).iloc[:, :3])
    print("\nTransformed sample (first 3 columns):")
    print(sample_data.iloc[:, :3])
    print("\nReconstructed sample (first 3 columns):")
    print(reconstructed_data.iloc[:, :3])

    unique_column_values = get_unique_column_values(data_numerical)

    with open(f'{folder_path}/unique_column_values.json', 'w') as f:
        json.dump(unique_column_values, f, indent=2)

    