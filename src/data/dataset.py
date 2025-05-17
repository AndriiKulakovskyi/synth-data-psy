import numpy as np
import os

# from src import tools
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat):
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index):
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]

        sample = (this_num, this_cat)

        return sample

    def __len__(self):
        return self.X_num.shape[0]

def preprocess(dataset_path, task_type = 'binclass', inverse = False, cat_encoding = None, concat = True):
    
    T_dict = {}

    T_dict['normalization'] = "quantile"
    T_dict['num_nan_policy'] = 'mean'
    T_dict['cat_nan_policy'] =  None
    T_dict['cat_min_frequency'] = None
    T_dict['cat_encoding'] = cat_encoding
    T_dict['y_policy'] = "default"

    T = tools.data.Transformations(**T_dict)

    dataset = make_dataset(data_path=dataset_path, T=T, task_type=task_type, change_val=False, concat=concat)

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']
        
        categories = tools.data.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)


        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform

            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        else:
            return X_num, X_cat, categories, d_numerical
    else:
        return dataset


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



def concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


# def make_dataset(data_path: str, T: tools.data.Transformations, task_type, change_val: bool, concat = True):

#     # classification
#     if task_type == 'binclass' or task_type == 'multiclass':
#         X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
#         X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
#         y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

#         for split in ['train', 'test']:
#             X_num_t, X_cat_t, y_t = tools.data.read_pure_data(data_path, split)
#             if X_num is not None:
#                 X_num[split] = X_num_t
#             if X_cat is not None:
#                 if concat:
#                     X_cat_t = concat_y_to_X(X_cat_t, y_t)
#                 X_cat[split] = X_cat_t  
#             if y is not None:
#                 y[split] = y_t
#     else:
#         # regression
#         X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
#         X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
#         y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

#         for split in ['train', 'test']:
#             X_num_t, X_cat_t, y_t = tools.data.read_pure_data(data_path, split)

#             if X_num is not None:
#                 if concat:
#                     X_num_t = concat_y_to_X(X_num_t, y_t)
#                 X_num[split] = X_num_t
#             if X_cat is not None:
#                 X_cat[split] = X_cat_t
#             if y is not None:
#                 y[split] = y_t

#     info = tools.data.load_json(os.path.join(data_path, 'info.json'))

#     D = tools.data.Dataset(
#         X_num,
#         X_cat,
#         y,
#         y_info={},
#         task_type=tools.data.TaskType(info['task_type']),
#         n_classes=info.get('n_classes')
#     )

#     if change_val:
#         D = tools.data.change_val(D)

#     # def categorical_to_idx(feature):
#     #     unique_categories = np.unique(feature)
#     #     idx_mapping = {category: index for index, category in enumerate(unique_categories)}
#     #     idx_feature = np.array([idx_mapping[category] for category in feature])
#     #     return idx_feature

#     # for split in ['train', 'val', 'test']:
#     # D.y[split] = categorical_to_idx(D.y[split].squeeze(1))

#     return tools.data.transform_dataset(D, T, None)


def preprocess_and_save(data_path: str, info: dict, name: str, train_ratio: float = 0.9, save_dir_base: str = 'data', synthetic_dir_base: str = 'synthetic'):
    # 1. Load data
    data_df = pd.read_csv(data_path, header=info.get('header', 'infer'))
    num_data = data_df.shape[0]
    column_names = info.get('column_names') or data_df.columns.tolist()
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    # 2. Column mapping
    def get_column_name_mapping(data_df: pd.DataFrame, num_col_idx: list, cat_col_idx: list, target_col_idx: list, column_names: list = None):
        if not column_names:
            column_names = np.array(data_df.columns.tolist())
        idx_mapping = {}
        curr_num_idx = 0
        curr_cat_idx = len(num_col_idx)
        curr_target_idx = curr_cat_idx + len(cat_col_idx)
        for idx in range(len(column_names)):
            if idx in num_col_idx:
                idx_mapping[int(idx)] = curr_num_idx
                curr_num_idx += 1
            elif idx in cat_col_idx:
                idx_mapping[int(idx)] = curr_cat_idx
                curr_cat_idx += 1
            else:
                idx_mapping[int(idx)] = curr_target_idx
                curr_target_idx += 1
        inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}
        idx_name_mapping = {i: column_names[i] for i in range(len(column_names))}
        return idx_mapping, inverse_idx_mapping, idx_name_mapping

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # 3. Train/test split
    num_train = int(num_data * train_ratio)
    num_test = num_data - num_train

    def train_val_test_split(df : pd.DataFrame, cat_columns: list, num_train: int, num_test: int, random_state: int = 42):
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        train_df = df_shuffled.iloc[:num_train].copy()
        test_df = df_shuffled.iloc[num_train:num_train + num_test].copy()
        seed = random_state
        return train_df, test_df, seed
    

    train_df, test_df, seed = train_val_test_split(data_df, cat_columns, num_train, num_test)
    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    # 4. Build col_info
    col_info = {}
    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'numerical'
        col_info[col_idx]['max'] = float(train_df[col_idx].max())
        col_info[col_idx]['min'] = float(train_df[col_idx].min())
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info[col_idx]['type'] = 'categorical'
        col_info[col_idx]['categorizes'] = list(set(train_df[col_idx]))
    for col_idx in target_col_idx:
        if info['task_type'] == 'regression':
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'numerical'
            col_info[col_idx]['max'] = float(train_df[col_idx].max())
            col_info[col_idx]['min'] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info[col_idx]['type'] = 'categorical'
            col_info[col_idx]['categorizes'] = list(set(train_df[col_idx]))
    info['column_info'] = col_info

    # 5. Rename columns to original names for saving
    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    # 6. Replace '?' with nan for numerical, 'nan' for categorical
    for col in num_columns:
        train_df.loc[train_df[col] == '?', col] = np.nan
        test_df.loc[test_df[col] == '?', col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == '?', col] = 'nan'
        test_df.loc[test_df[col] == '?', col] = 'nan'

    # 7. Extract arrays
    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy()
    y_train = train_df[target_columns].to_numpy()
    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy()
    y_test = test_df[target_columns].to_numpy()

    # 8. Save arrays and CSVs
    save_dir = os.path.join(save_dir_base, name)
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/X_num_train.npy', X_num_train)
    np.save(f'{save_dir}/X_cat_train.npy', X_cat_train)
    np.save(f'{save_dir}/y_train.npy', y_train)
    np.save(f'{save_dir}/X_num_test.npy', X_num_test)
    np.save(f'{save_dir}/X_cat_test.npy', X_cat_test)
    np.save(f'{save_dir}/y_test.npy', y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)
    train_df.to_csv(f'{save_dir}/train.csv', index=False)
    test_df.to_csv(f'{save_dir}/test.csv', index=False)

    synthetic_dir = os.path.join(synthetic_dir_base, name)
    os.makedirs(synthetic_dir, exist_ok=True)
    train_df.to_csv(f'{synthetic_dir}/real.csv', index=False)
    test_df.to_csv(f'{synthetic_dir}/test.csv', index=False)

    # 9. Update info
    info['column_names'] = column_names
    info['train_num'] = train_df.shape[0]
    info['test_num'] = test_df.shape[0]
    info['idx_mapping'] = idx_mapping
    info['inverse_idx_mapping'] = inverse_idx_mapping
    info['idx_name_mapping'] = idx_name_mapping

    # 10. Metadata
    metadata = {'columns': {}}
    task_type = info['task_type']
    for i in num_col_idx:
        metadata['columns'][i] = {'sdtype': 'numerical', 'computer_representation': 'Float'}
    for i in cat_col_idx:
        metadata['columns'][i] = {'sdtype': 'categorical'}
    if task_type == 'regression':
        for i in target_col_idx:
            metadata['columns'][i] = {'sdtype': 'numerical', 'computer_representation': 'Float'}
    else:
        for i in target_col_idx:
            metadata['columns'][i] = {'sdtype': 'categorical'}
    info['metadata'] = metadata

    # 11. Save info
    with open(f'{save_dir}/info.json', 'w') as file:
        json.dump(info, file, indent=4)

    # 12. Print summary
    print(f'Processing and Saving {name} Successfully!')
    print(name)
    print('Total', info['train_num'] + info['test_num'])
    print('Train', info['train_num'])
    print('Test', info['test_num'])
    if info['task_type'] == 'regression':
        num = len(info['num_col_idx'] + info['target_col_idx'])
        cat = len(info['cat_col_idx'])
    else:
        cat = len(info['cat_col_idx'] + info['target_col_idx'])
        num = len(info['num_col_idx'])
    print('Num', num)
    print('Cat', cat)
    print('Numerical', X_num_train.shape)
    print('Categorical', X_cat_train.shape)
