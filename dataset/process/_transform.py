

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import sys
import tqdm
import _utils as dutils
import datetime
import matplotlib.pyplot as plt
import os
import ast
import html
import re


def parse_item_fields(fields_arg):
    if not fields_arg:
        return ['categories']

    fields = [field.strip() for field in fields_arg.split(',') if field.strip()]
    if not fields:
        return ['categories']

    deduped_fields = []
    seen = set()
    for field in fields:
        if field not in seen:
            deduped_fields.append(field)
            seen.add(field)
    return deduped_fields


def format_metadata_value(field_name, value):
    if value is None:
        return ''

    if field_name == 'categories':
        return format_categories(value)

    if isinstance(value, (list, tuple)):
        cleaned = [clean_html_text(v) for v in value]
        cleaned = [v for v in cleaned if v]
        return ' | '.join(cleaned)

    if isinstance(value, dict):
        try:
            return clean_html_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except Exception:
            return clean_html_text(str(value))

    return clean_html_text(value)

def decompress_gz_file(gz_file_path, output_path):
    '''
    Decompress .gz file to specified output path
    '''
    print(f'Decompressing {gz_file_path}...')
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.writelines(f_in)
    print(f'Decompressed to {output_path}')

def clean_html_text(value):
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\xa0', ' ')
    # replace double quotes to avoid CSV quoting issues, remove tabs/newlines
    text = text.replace('\"', "'")
    text = text.replace('"', "'")
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_categories(categories):
    if not categories:
        return ''

    parts = []
    if isinstance(categories, str):
        cleaned = clean_html_text(categories)
        if cleaned:
            parts.append(cleaned)
    else:
        for category_path in categories:
            if isinstance(category_path, (list, tuple)):
                for part in category_path:
                    cleaned_part = clean_html_text(part)
                    if cleaned_part:
                        parts.append(cleaned_part)
            else:
                cleaned_value = clean_html_text(category_path)
                if cleaned_value:
                    parts.append(cleaned_value)

    # return items wrapped in single quotes and comma-separated
    return ', '.join([f"'{p}'" for p in parts])

def load_item_metadata(dataset_name, item_fields=None):
    '''
    Load item metadata from meta_{dataset_name}.json
    Returns dict: item_id -> {title, brand, price, ...}
    '''
    item_metadata = {}
    item_fields = item_fields or ['categories']
    raw_dir = './raw'
    meta_json_file = os.path.join(raw_dir, 'meta_{}.json'.format(dataset_name))
    meta_gz_file = os.path.join(raw_dir, 'meta_{}.json.gz'.format(dataset_name))
    
    # Check if .json file exists, if not check and decompress .gz file
    if not os.path.exists(meta_json_file):
        if os.path.exists(meta_gz_file):
            decompress_gz_file(meta_gz_file, meta_json_file)
        else:
            print(f'Warning: Metadata file not found ({meta_json_file} or {meta_gz_file})')
            return item_metadata
    
    print(f'Loading metadata from {meta_json_file}...')
    try:
        with open(meta_json_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f'Error opening metadata file: {e}')
        return item_metadata
    
    loaded_count = 0
    error_count = 0
    
    for line_num, line in enumerate(tqdm.tqdm(lines), 1):
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        try:
            # Try JSON format first
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                # If JSON fails, try Python dict format (with single quotes)
                meta = ast.literal_eval(line)
            
            item_id = clean_html_text(meta.get('asin', ''))
            if item_id:
                item_metadata[item_id] = {
                    field: format_metadata_value(field, meta.get(field, ''))
                    for field in item_fields
                }
                loaded_count += 1
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            error_count += 1
            if error_count <= 3:
                print(f'\nLine {line_num}: Parse error: {e}')
                print(f'  Content preview: {line[:100]}...')
            continue
        except Exception as e:
            error_count += 1
            if error_count <= 3:
                print(f'\nLine {line_num}: Error: {e}')
            continue
    
    if error_count > 3:
        print(f'Skipped {error_count} malformed lines in total')
    print(f'Successfully loaded metadata for {loaded_count} items')
    return item_metadata

# return (user item rating timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score, item_metadata=None):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    
    raw_dir = './raw'
    json_file = os.path.join(raw_dir, 'reviews_{}_5.json'.format(dataset_name))
    gz_file = os.path.join(raw_dir, 'reviews_{}_5.json.gz'.format(dataset_name))
    
    # Check if .json file exists, if not check and decompress .gz file
    if not os.path.exists(json_file):
        if os.path.exists(gz_file):
            decompress_gz_file(gz_file, json_file)
        else:
            raise FileNotFoundError(f'Neither {json_file} nor {gz_file} found!')
    
    lines = open(json_file).readlines()
    for line in tqdm.tqdm(lines):
        inter = json.loads(line.strip())
        if float(inter['overall']) <= rating_score: # Less than a certain percentage.
            continue
        user = inter['reviewerID']
        item = inter['asin']
        rating = float(inter['overall'])
        time = inter['unixReviewTime']
        datas.append((user, item, rating, int(time)))

    return datas

def process(data_name='Beauty', item_fields=None):
    item_fields = item_fields or ['categories']
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    
    # Load metadata from file first
    print(f'Loading metadata for {data_name}...')
    item_metadata = load_item_metadata(data_name, item_fields=item_fields)
    
    # Dataset name mapping from full name to directory name
    dataset_dir_map = {
        'Beauty': 'Beauty',
        'Sports_and_Outdoors': 'Sports',
        'Toys_and_Games': 'Toys'
    }

    if data_name in ["Sports_and_Outdoors", "Toys_and_Games", "Beauty"]:
        datas = Amazon(data_name, rating_score)
    
    # Convert datas format for get_interaction compatibility (need to remove rating temporarily)
    datas_no_rating = [(user, item, timestamp) for user, item, rating, timestamp in datas]
    interactions_with_rating = datas  # Keep original with rating

    user_items, time_interval = dutils.get_interaction(datas_no_rating, data_name)
    print(f'{data_name} Raw data has been processed!')
    # raw_id user: [item1, item2, item3...]
    user_items, time_interval = dutils.filter_Kcore(user_items, time_interval, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    user_items, time_interval, user_num, item_num, data_maps = dutils.id_map(user_items, time_interval)  # new_num_id

    avg_seqlen = np.mean([len(seq) for seq in user_items.values()])
    user_count, item_count, _ = dutils.check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())

    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'\n====={data_name}=====\n' + \
                f'Total User: {user_num}, Avg User: {user_avg:.2f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.2f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Avg Sequence Length: {avg_seqlen:.1f}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    # Create output directory if it doesn't exist
    output_dir = dataset_dir_map[data_name]
    output_dir_path = f'../{output_dir}'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    # Extract id mappings for easier access
    user2id = data_maps['user2id']
    item2id = data_maps['item2id']
    id2user = data_maps['id2user']
    id2item = data_maps['id2item']
    
    # 1. Save user id mapping JSON
    user_map_file = f'../{output_dir}/{output_dir}_user_id_map.json'
    user_map = {
        'raw2id': user2id,
        'id2raw': {numeric_id: id2user[numeric_id] for numeric_id in sorted(id2user.keys(), key=lambda x: int(x))}
    }
    with open(user_map_file, 'w', encoding='utf-8') as out:
        json.dump(user_map, out, ensure_ascii=False, indent=2)
    print(f'User id mapping saved to {user_map_file}')
    
    # 2. Save item id mapping JSON
    item_map_file = f'../{output_dir}/{output_dir}_item_id_map.json'
    item_map = {
        'raw2id': item2id,
        'id2raw': {numeric_id: id2item[numeric_id] for numeric_id in sorted(id2item.keys(), key=lambda x: int(x))}
    }
    with open(item_map_file, 'w', encoding='utf-8') as out:
        json.dump(item_map, out, ensure_ascii=False, indent=2)
    print(f'Item id mapping saved to {item_map_file}')
    
    # 3. Save interactions in .inter format
    inter_file = f'../{output_dir}/{output_dir}.inter'
    with open(inter_file, 'w') as out:
        out.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        # Build id mapping dicts for O(1) lookup
        raw_user2numeric = {v: k for k, v in id2user.items()}
        raw_item2numeric = {v: k for k, v in id2item.items()}
        
        # Directly iterate through interactions and write
        for user, item, rating, timestamp in interactions_with_rating:
            if user in raw_user2numeric and item in raw_item2numeric:
                user_numeric_id = raw_user2numeric[user]
                item_numeric_id = raw_item2numeric[item]
                out.write(f'{user_numeric_id}\t{item_numeric_id}\t{rating}\t{timestamp}\n')
    print(f'Interactions saved to {inter_file}')

    # 4. Save item side-feature file in .item format
    item_file = f'../{output_dir}/{output_dir}.item'
    item_header = ['item_id:token'] + [f"{field}:token_seq" if field == 'categories' else f"{field}:token" for field in item_fields]
    with open(item_file, 'w', encoding='utf-8') as out:
        out.write('\t'.join(item_header) + '\n')
        for numeric_id in sorted(id2item.keys(), key=lambda x: int(x)):
            original_item_id = id2item[numeric_id]
            metadata = item_metadata.get(original_item_id, {})
            row = [numeric_id]
            for field in item_fields:
                field_value = clean_html_text(metadata.get(field, ''))
                row.append(field_value)
            out.write('\t'.join(row) + '\n')
    print(f'Item side-feature saved to {item_file}')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python _transform.py <dataset_name|all> [item_fields]")
        print("Example: python _transform.py Beauty categories,title")
        print("Default item_fields: categories")
        sys.exit(1)

    dataname = sys.argv[1]
    item_fields = parse_item_fields(sys.argv[2] if len(sys.argv) >= 3 else None)
    available_datasets = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games']
    if dataname == 'all':
        for name in available_datasets:
            process(name, item_fields=item_fields)
    elif dataname in available_datasets:
        process(dataname, item_fields=item_fields)
    else:
        print('Invalid dataset name')
        print(f"Available datasets: {available_datasets}")
        print("To transform all datasets at once, enter 'all' as the dataset name.")