# prepare_aihub.py

import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_category_from_path(path: str) -> str:
    """파일 경로에서 문서 카테고리를 추출합니다."""
    if '법률' in path:
        return '법률'
    if '사설' in path:
        return '사설'
    if '신문기사' in path:
        return '신문기사'
    return 'etc'

def process_folder(folder_path: str) -> pd.DataFrame:
    """지정된 폴더의 모든 JSON 파일을 처리하여 DataFrame으로 반환합니다."""
    json_files = glob(os.path.join(folder_path, '**/*.json'), recursive=True)
    
    if not json_files:
        print(f"경고: {folder_path}에서 JSON 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
        
    folder_name = os.path.basename(folder_path)
    print(f"{folder_name}에서 {len(json_files)}개의 JSON 파일을 찾았습니다. 처리 중...")
    
    data = []
    for file_path in tqdm(json_files, desc=f"{folder_name} 처리 중"):
        category = get_category_from_path(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for doc in json_data['documents']:
                    # text는 문장 리스트이므로 join하여 합쳐줍니다.
                    dialogue = "\n".join([item['sentence'] for sublist in doc['text'] for item in sublist])
                    summary = doc['abstractive'][0]
                    
                    data.append({
                        "fname": doc['id'],
                        "dialogue": dialogue,
                        "summary": summary,
                        "category": category
                    })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"파일 처리 오류 {file_path}: {e}")
            
    return pd.DataFrame(data)

def create_datasets(base_data_path: str):
    """
    AI Hub 데이터셋 폴더에서 JSON 파일들을 읽어 train, validation, test CSV로 변환합니다.
    - Training 폴더 -> train.csv
    - Validation 폴더 -> val.csv (50%) / test.csv (50%), 카테고리별 균등 분할
    """
    train_path = os.path.join(base_data_path, 'Training')
    val_path = os.path.join(base_data_path, 'Validation')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"오류: {base_data_path} 경로에 'Training' 또는 'Validation' 폴더가 없습니다.")
        return

    # 1. Training 데이터 처리
    train_df = process_folder(train_path)
    
    # 2. Validation 데이터 처리
    validation_df = process_folder(val_path)

    if validation_df.empty:
        print("Validation 데이터가 비어있어 val.csv와 test.csv를 생성할 수 없습니다.")
        # train.csv만 있는 경우를 위해 저장은 진행
        if not train_df.empty:
            output_dir = 'data'
            os.makedirs(output_dir, exist_ok=True)
            train_output_path = os.path.join(output_dir, 'train.csv')
            if 'category' in train_df.columns:
                train_df.drop(columns=['category'], inplace=True)
            train_df.to_csv(train_output_path, index=False, encoding='utf-8')
            print(f"\n데이터셋 생성을 완료했습니다.")
            print(f"학습 데이터 저장 위치: {train_output_path} ({len(train_df)}개 샘플)")
        return

    # 3. Validation 데이터를 val/test로 50:50 분할 (카테고리 기준)
    val_df, test_df = train_test_split(
        validation_df,
        test_size=0.5,
        random_state=42,
        stratify=validation_df['category']
    )

    # CSV 저장을 위해 data 폴더 생성
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'train.csv')
    val_output_path = os.path.join(output_dir, 'val.csv')
    test_output_path = os.path.join(output_dir, 'test.csv')

    # 분할에 사용된 category 컬럼은 최종 파일에서 제외
    if 'category' in train_df.columns:
        train_df.drop(columns=['category'], inplace=True)
    if 'category' in val_df.columns:
        val_df.drop(columns=['category'], inplace=True)
    if 'category' in test_df.columns:
        test_df.drop(columns=['category'], inplace=True)

    train_df.to_csv(train_output_path, index=False, encoding='utf-8')
    val_df.to_csv(val_output_path, index=False, encoding='utf-8')
    test_df.to_csv(test_output_path, index=False, encoding='utf-8')
    
    print(f"\n데이터셋 생성을 완료했습니다.")
    print(f"학습 데이터 저장 위치: {train_output_path} ({len(train_df)}개 샘플)")
    print(f"검증 데이터 저장 위치: {val_output_path} ({len(val_df)}개 샘플)")
    print(f"테스트 데이터 저장 위치: {test_output_path} ({len(test_df)}개 샘플)")


if __name__ == "__main__":
    # 'Training'과 'Validation' 폴더가 들어있는 상위 폴더 경로
    YOUR_AIHUB_DATA_PATH = "data"
    
    if not os.path.exists(YOUR_AIHUB_DATA_PATH):
        print(f"경로를 찾을 수 없습니다: {YOUR_AIHUB_DATA_PATH}")
    else:
        create_datasets(YOUR_AIHUB_DATA_PATH)