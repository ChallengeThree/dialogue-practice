# prepare_xsum.py

import os
from datasets import load_dataset
import pandas as pd

def convert_xsum_to_csv(num_samples: int = 100):
    """
    XSum 데이터셋을 로드하여 'dialogue', 'summary' 형식의 CSV로 변환합니다.
    """
    print("Loading XSum dataset from Hugging Face...")
    # 'validation' 스플릿에서 num_samples 만큼만 가져옵니다.
    xsum_dataset = load_dataset("xsum", split=f"validation[:{num_samples}]")
    
    print("Converting dataset to pandas DataFrame...")
    data = []
    for i, example in enumerate(xsum_dataset):
        data.append({
            "fname": f"xsum_{i}",
            # 대회 데이터 형식에 맞게 컬럼 이름 변경
            "dialogue": example["document"],
            "summary": example["summary"]
        })
        
    df = pd.DataFrame(data)
    
    # data 폴더에 저장
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'xsum_eval.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\nSuccessfully converted {num_samples} samples.")
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    convert_xsum_to_csv()