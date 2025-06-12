import yaml
from src.data.processing import *
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config('config/config.yaml')

    folder_path = config['data']['save_folder']
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    threshold_nan_columns = config['data']['threshold_nan_columns']
    threshold_nan_rows = config['data']['threshold_nan_rows']
    numerical_min_conversion_rate = config['data']['numerical_min_conversion_rate']
    categorical_threshold = config['data']['categorical_threshold']
    save_transforms = config['data']['save_transforms']
    
    raw_data = load_data('FACE/neuropsy_v0.csv', config['data']['drop_columns'])

    data_cleaned = drop_cols_by_missing_count(raw_data, threshold=threshold_nan_columns)
    data_cleaned = drop_rows_by_missing_count(data_cleaned, threshold=threshold_nan_rows)
    data_numerical = convert_to_numerical(data_cleaned, min_conversion_rate=numerical_min_conversion_rate)
    
    data_transformer = DataTransformer(categorical_threshold=categorical_threshold, save_transforms=save_transforms, save_folder=folder_path)
    
    # Split data into numerical and categorical components
    data_ml_ready = data_transformer.fit_transform(data_numerical)
    numerical_df, categorical_df, categorical_info_df = split_ml_data(data_ml_ready, data_transformer)
    
    # Impute remaining missing data for numerical and categorical columns
    numerical_imputed, categorical_imputed, imputation_info = impute_missing_data(
        numerical_df, categorical_df, 
        numerical_strategy=config['data']['imputation']['numerical_strategy'], 
        categorical_strategy=config['data']['imputation']['categorical_strategy'],
        knn_neighbors=config['data']['imputation']['knn_neighbors'],
        save_folder=folder_path
    )
    
    # Save imputed datasets
    if not numerical_imputed.empty:
        numerical_imputed.to_csv(f'{folder_path}/numerical_data_imputed.csv', sep=';', index=False)
        print(f"Imputed numerical data saved to '{folder_path}/numerical_data_imputed.csv'")
    
    if not categorical_imputed.empty:
        categorical_imputed.to_csv(f'{folder_path}/categorical_data_imputed.csv', sep=';', index=False)
        print(f"Imputed categorical data saved to '{folder_path}/categorical_data_imputed.csv'")
    
    # Save categorical info
    if not categorical_info_df.empty:
        categorical_info_df.to_csv(f'{folder_path}/categorical_info.csv', sep=';', index=False)
        print(f"Categorical info saved to '{folder_path}/categorical_info.csv'")
    
    # Stack numerical and categorical data and plot correlation heatmap
    data_imputed = pd.concat([numerical_imputed, categorical_imputed], axis=1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(data_imputed.corr(), annot=False, cmap='viridis')
    # add line to the plot to separate numerical and categorical columns
    plt.axvline(x=numerical_imputed.shape[1], color='black', linewidth=1)
    plt.axhline(y=numerical_imputed.shape[1], color='black', linewidth=1)
    plt.savefig(f'{folder_path}/correlation_heatmap.png')