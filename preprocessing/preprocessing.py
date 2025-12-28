#!/usr/bin/env python3
"""
Multi-modal Recommendation Data Preprocessing Pipeline
Processes Amazon dataset to extract U-I interactions, perform 5-core filtering, 
feature extraction and generate user-user matrix for DualGNN.
"""

import os
import csv
import pandas as pd
import numpy as np
import argparse
import yaml
from collections import Counter, defaultdict
import gzip
import json
import array
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def setup_paths(dataset_name='Games', method=None):
    """Setup dataset paths and ensure directories exist"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Original dataset path for reading source files
    source_dataset_path = os.path.join(base_path, '..', 'dataset', dataset_name)
    
    # Target dataset path for saving processed files
    if method and method != 'default':
        target_dataset_path = os.path.join(base_path, '..', 'data', f'{dataset_name}-{method}')
    else:
        target_dataset_path = source_dataset_path
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dataset_path):
        os.makedirs(target_dataset_path, exist_ok=True)
    
    return source_dataset_path, target_dataset_path


def load_config(dataset_name):
    """Load configuration from yaml files"""
    config = {}
    base_path = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(base_path, '..', 'src')
    config_dir = os.path.join(src_path, 'configs')
    
    overall_config_file = os.path.join(config_dir, "overall.yaml")
    dataset_config_file = os.path.join(config_dir, "dataset", f"{dataset_name.split('-')[0]}.yaml")

    for file in [overall_config_file, dataset_config_file]:
        if os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                tmp_d = yaml.safe_load(f)
                config.update(tmp_d)
    
    return config


def perform_k_core_filtering(df, min_u_num=5, min_i_num=5):
    """Perform k-core filtering on the dataset"""
    learner_id, course_id = 'userID', 'itemID'
    
    def get_illegal_ids_by_inter_num(df, field, min_num=None):
        if field is None or min_num is None:
            return set()
        
        ids = df[field].values
        inter_num = Counter(ids)
        illegal_ids = {id_ for id_ in inter_num if inter_num[id_] < min_num}
        print(f'{len(illegal_ids)} illegal_ids_by_inter_num, field={field}')
        return illegal_ids

    def filter_by_k_core():
        while True:
            ban_users = get_illegal_ids_by_inter_num(df, field=learner_id, min_num=min_u_num)
            ban_items = get_illegal_ids_by_inter_num(df, field=course_id, min_num=min_i_num)
            
            if len(ban_users) == 0 and len(ban_items) == 0:
                return
            
            dropped_inter = pd.Series(False, index=df.index)
            if learner_id:
                dropped_inter |= df[learner_id].isin(ban_users)
            if course_id:
                dropped_inter |= df[course_id].isin(ban_items)
            
            print(f'{len(dropped_inter)} dropped interactions')
            df.drop(df.index[dropped_inter], inplace=True)

    filter_by_k_core()
    return df

def process_ratings_data(source_path, target_path, dataset_name='Games'):
    """Load and preprocess ratings data"""
    ratings_file = os.path.join(source_path, f'ratings_{dataset_name}.csv')
    df = pd.read_csv(ratings_file, names=['userID', 'itemID', 'rating', 'timestamp'], header=None)
    print(f'Initial shape: {df.shape}')
    
    # Clean data
    df.dropna(subset=['userID', 'itemID', 'timestamp'], inplace=True)
    df.drop_duplicates(subset=['userID', 'itemID', 'timestamp'], inplace=True)
    print(f'After cleaning: {df.shape}')
    
    # Apply k-core filtering
    df = perform_k_core_filtering(df)
    print(f'After k-core filtering: {df.shape}')
    
    return df


def reindex_data(df, target_path):
    """Reindex users and items starting from 0"""
    df.reset_index(drop=True, inplace=True)
    
    uid_field, iid_field = 'userID', 'itemID'
    uni_users = pd.unique(df[uid_field])
    uni_items = pd.unique(df[iid_field])
    
    # Create mappings starting from 0
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}
    
    # Apply mappings
    df[uid_field] = df[uid_field].map(u_id_map)
    df[iid_field] = df[iid_field].map(i_id_map)
    df[uid_field] = df[uid_field].astype(int)
    df[iid_field] = df[iid_field].astype(int)
    
    # Save mapping files
    u_df = pd.DataFrame(list(u_id_map.items()), columns=['user_id', 'userID'])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['asin', 'itemID'])
    
    u_df.to_csv(os.path.join(target_path, 'u_id_mapping.csv'), sep='\t', index=False)
    i_df.to_csv(os.path.join(target_path, 'i_id_mapping.csv'), sep='\t', index=False)
    print('Mapping files saved')
    
    return df


def split_data(df, target_path, dataset_name='Games', splitting=[0.8, 0.1, 0.1]):
    """Split data into train/validation/test sets"""
    tot_ratio = sum(splitting)
    ratios = [i for i in splitting if i > 0.0]
    ratios = [_ / tot_ratio for _ in ratios]
    split_ratios = np.cumsum(ratios)[:-1]
    
    ts_id = 'timestamp'
    split_timestamps = list(np.quantile(df[ts_id], split_ratios))
    
    # Create splits
    df_train = df.loc[df[ts_id] < split_timestamps[0]].copy()
    df_val = df.loc[(split_timestamps[0] <= df[ts_id]) & (df[ts_id] < split_timestamps[1])].copy()
    df_test = df.loc[(split_timestamps[1] <= df[ts_id])].copy()
    
    # Add labels
    x_label = 'x_label'
    df_train[x_label] = 0
    df_val[x_label] = 1
    df_test[x_label] = 2
    
    # Combine and save
    temp_df = pd.concat([df_train, df_val, df_test])
    temp_df = temp_df[['userID', 'itemID', 'rating', ts_id, x_label]]
    
    rslt_file = f'{dataset_name}-indexed.inter'
    temp_df.to_csv(os.path.join(target_path, rslt_file), sep='\t', index=False)
    print(f'Split data saved to {rslt_file}')
    
    return temp_df


def split_data_v4(df, target_path, dataset_name='Games'):
    """Alternative splitting method (version 4)"""
    df = df.sample(frac=1).reset_index(drop=True)
    df.sort_values(by=['userID'], inplace=True)
    
    uid_field, iid_field = 'userID', 'itemID'
    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {u: list(u_ls) for u, u_ls in uid_freq}
    
    new_label = []
    u_ids_sorted = sorted(u_i_dict.keys())
    
    for u in u_ids_sorted:
        items = u_i_dict[u]
        n_items = len(items)
        
        if n_items < 10:
            tmp_ls = [0] * (n_items - 2) + [1] + [2]
        else:
            val_test_len = int(n_items * 0.2)
            train_len = n_items - val_test_len
            val_len = val_test_len // 2
            test_len = val_test_len - val_len
            tmp_ls = [0] * train_len + [1] * val_len + [2] * test_len
        
        new_label.extend(tmp_ls)
    
    df['x_label'] = new_label
    
    new_labeled_file = f'{dataset_name}-indexed-v4.inter'
    df.to_csv(os.path.join(target_path, new_labeled_file), sep='\t', index=False)
    print(f'V4 split data saved to {new_labeled_file}')
    
    return df






def process_metadata(source_path, target_path, dataset_name='Games'):
    """Process and reindex metadata features"""
    # Load item mapping from target path
    i_id_mapping_file = os.path.join(target_path, 'i_id_mapping.csv')
    mapping_df = pd.read_csv(i_id_mapping_file, sep='\t')
    print(f'Mapping shape: {mapping_df.shape}')
    
    # Load metadata from source path
    meta_file = os.path.join(source_path, f'meta_{dataset_name}.json.gz')
    
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)
    
    def getDF(path):
        i = 0
        df = {}
        for d in parse(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')
    
    meta_df = getDF(meta_file)
    print(f'Metadata shape: {meta_df.shape}')
    
    # Remap item IDs
    map_dict = dict(zip(mapping_df['asin'], mapping_df['itemID']))
    meta_df['itemID'] = meta_df['asin'].map(map_dict)
    meta_df.dropna(subset=['itemID'], inplace=True)
    meta_df['itemID'] = meta_df['itemID'].astype('int64')
    meta_df.sort_values(by=['itemID'], inplace=True)
    
    # Reorder columns
    ori_cols = meta_df.columns.tolist()
    ret_cols = [ori_cols[-1]] + ori_cols[:-1]  # Move itemID to first
    ret_df = meta_df[ret_cols]
    
    # Save processed metadata to target path
    output_file = os.path.join(target_path, f'meta-{dataset_name}.csv')
    ret_df.to_csv(output_file, index=False)
    print(f'Processed metadata saved to {output_file}')
    
    return ret_df


def extract_text_features(target_path, dataset_name='Games', model_name='all-MiniLM-L6-v2'):
    """Extract text features using sentence transformers"""
    meta_file = os.path.join(target_path, f'meta-{dataset_name}.csv')
    df = pd.read_csv(meta_file)
    df.sort_values(by=['itemID'], inplace=True)
    
    print(f'Extracting text features for {df.shape[0]} items')
    
    # Fill missing values
    df['description'] = df['description'].fillna(" ")
    df['title'] = df['title'].fillna(" ")
    df['brand'] = df['brand'].fillna(" ")
    df['categories'] = df['categories'].fillna(" ")
    
    # Create sentences combining multiple fields
    sentences = []
    for i, row in df.iterrows():
        sen = row['title'] + ' ' + row['brand'] + ' '
        
        # Process categories
        try:
            cates = eval(row['categories'])
            if isinstance(cates, list) and len(cates) > 0 and isinstance(cates[0], list):
                for c in cates[0]:
                    sen = sen + c + ' '
        except:
            pass
        
        sen += row['description']
        sen = sen.replace('\n', ' ')
        sentences.append(sen)
    
    # Encode sentences
    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)
    print('Text encoding completed!')
    
    # Verify and save
    assert sentence_embeddings.shape[0] == df.shape[0]
    output_file = os.path.join(target_path, 'text_feat.npy')
    np.save(output_file, sentence_embeddings)
    print(f'Text features saved to {output_file}')
    
    return sentence_embeddings


def extract_image_features(source_path, target_path, dataset_name='Games'):
    """Extract image features from binary file"""
    def readImageFeatures(path):
        f = open(path, 'rb')
        while True:
            asin = f.read(10).decode('UTF-8')
            if asin == '': 
                break
            a = array.array('f')
            a.fromfile(f, 4096)
            yield asin, a.tolist()
    
    # Load metadata for mapping from target path
    meta_file = os.path.join(target_path, f'meta-{dataset_name}.csv')
    df = pd.read_csv(meta_file)
    item2id = dict(zip(df['asin'], df['itemID']))
    
    # Process image features from source path
    img_file = os.path.join(source_path, f'image_features_{dataset_name}.b')
    img_data = readImageFeatures(img_file)
    
    feats = {}
    avg = []
    
    for d in img_data:
        if d[0] in item2id:
            feats[int(item2id[d[0]])] = d[1]
            avg.append(d[1])
    
    avg = np.array(avg).mean(0).tolist()
    
    # Fill missing features with average
    ret = []
    non_no = []
    for i in range(len(item2id)):
        if i in feats:
            ret.append(feats[i])
        else:
            non_no.append(i)
            ret.append(avg)
    
    print(f'Items missing image features: {len(non_no)}')
    assert len(ret) == len(item2id)
    
    # Save features to target path
    np.save(os.path.join(target_path, 'image_feat.npy'), np.array(ret))
    np.savetxt(os.path.join(target_path, "missed_img_itemIDs.csv"), non_no, delimiter=",", fmt='%d')
    print('Image features saved!')
    
    return np.array(ret)


def gen_user_matrix(all_edge, no_users):
    """Generate user-user interaction matrix for DualGNN"""
    edge_dict = defaultdict(set)
    
    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item)
    
    min_user = 0
    num_user = no_users
    user_graph_matrix = torch.zeros(num_user, num_user)
    key_list = list(edge_dict.keys())
    key_list.sort()
    
    bar = tqdm(total=len(key_list), desc="Generating user-user matrix")
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head+1, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            inter_len = len(item_head.intersection(item_rear))
            
            if inter_len > 0:
                user_graph_matrix[head_key-min_user][rear_key-min_user] = inter_len
                user_graph_matrix[rear_key-min_user][head_key-min_user] = inter_len
    
    bar.close()
    return user_graph_matrix


def generate_user_graph_dict(target_path, dataset_name):
    """Generate user graph dictionary for DualGNN model"""
    config = load_config(dataset_name)
    
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    inter_file = config['inter_file_name']
    
    # Load interaction data from target path
    train_df = pd.read_csv(os.path.join(target_path, inter_file), sep='\t')
    num_user = len(pd.unique(train_df[uid_field]))
    train_df = train_df[train_df['x_label'] == 0].copy()  # Only training data
    train_data = train_df[[uid_field, iid_field]].to_numpy()
    
    # Generate user-user matrix
    user_graph_matrix = gen_user_matrix(train_data, num_user)
    user_graph = user_graph_matrix
    user_num = torch.zeros(num_user)
    
    # Calculate number of connections for each user
    for i in range(num_user):
        user_num[i] = len(torch.nonzero(user_graph[i]))
        print(f"User {i}: {user_num[i]} connections")
    
    # Build user graph dictionary
    user_graph_dict = {}
    for i in range(num_user):
        if user_num[i] <= 200:
            user_i = torch.topk(user_graph[i], int(user_num[i]))
        else:
            user_i = torch.topk(user_graph[i], 200)
        
        edge_list_i = user_i.indices.numpy().tolist()
        edge_list_j = user_i.values.numpy().tolist()
        edge_list = [edge_list_i, edge_list_j]
        user_graph_dict[i] = edge_list
    
    # Save user graph dictionary to target path
    output_file = os.path.join(target_path, config['user_graph_dict_file'])
    np.save(output_file, user_graph_dict, allow_pickle=True)
    print(f'User graph dictionary saved to {output_file}')
    
    return user_graph_dict


def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='Multi-modal Recommendation Data Preprocessing')
    parser.add_argument('--dataset', '-d', type=str, default='Games', help='Dataset name')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'ratings', 'split', 'metadata', 'text', 'image', 'mean', 'usergraph'],
                       help='Which preprocessing step to run')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    method = "baseline"
    
    print(f'Starting preprocessing for dataset: {dataset_name}')
    if method:
        print(f'Using method: {method}')
    
    source_path, target_path = setup_paths(dataset_name, method)
    print(f'Source path (for reading raw data): {source_path}')
    print(f'Target path (for saving processed data): {target_path}')
    
    if args.step in ['all', 'ratings']:
        print('\n=== Step 1: Processing ratings data ===')
        df = process_ratings_data(source_path, target_path, dataset_name)
        df = reindex_data(df, target_path)
        df = split_data(df, target_path, dataset_name)
    
    if args.step in ['all', 'split']:
        print('\n=== Step 2: Alternative data splitting ===')
        inter_file = os.path.join(target_path, f'{dataset_name}-indexed.inter')
        if os.path.exists(inter_file):
            df = pd.read_csv(inter_file, sep='\t')
            df = split_data_v4(df, target_path, dataset_name)
    
    if args.step in ['all', 'metadata']:
        print('\n=== Step 3: Processing metadata ===')
        meta_df = process_metadata(source_path, target_path, dataset_name)
    
    # Feature extraction based on method
    if args.step in ['all', 'text', 'image']:
        if args.step in ['all', 'text']:
            print('\n=== Step 4: Extracting text features ===')
            text_features = extract_text_features(target_path, dataset_name)
        
        if args.step in ['all', 'image']:
            print('\n=== Step 5: Extracting image features ===')
            image_features = extract_image_features(source_path, target_path, dataset_name)
    
    if args.step in ['all', 'usergraph']:
        print('\n=== Step 7: Generating user graph ===')
        user_graph_dict = generate_user_graph_dict(target_path, dataset_name)
    
    print(f'\nPreprocessing completed for {dataset_name}!')
    if method:
        print(f'Features generated using method: {method}')
        print(f'Results saved in: {target_path}')
    else:
        print(f'Results saved in: {source_path}')


if __name__ == '__main__':
    main()
