# inference.py

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.utils import load_config
from src.data_loader import DataPreprocessor, SummarizationDataset

def inference():
    # 설정 파일 로드
    config_path = "config/config.yaml"
    config = load_config(config_path)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = config['inference']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please train a model first.")

    # 저장된 체크포인트에서 모델과 토크나이저를 모두 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    preprocessor = DataPreprocessor(config, tokenizer)
    test_path = os.path.join(config['data_dir'], 'test.csv')
    enc_test, fnames = preprocessor.prepare_data(test_path)
    tokenized_test = preprocessor.tokenize_data(enc_test)
    test_dataset = SummarizationDataset(tokenized_test, len(enc_test))
    test_dataloader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'])

    # 요약 생성
    summaries = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating summaries"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config['model']['decoder_max_len'],
                num_beams=config['inference']['num_beams'],
                no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                early_stopping=config['inference']['early_stopping']
            )
            
            decoded_summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            summaries.extend(decoded_summaries)

    # 불필요한 토큰 정리
    remove_tokens = config['model']['special_tokens'] + config['inference']['remove_tokens']
    cleaned_summaries = []
    for s in summaries:
        for token in remove_tokens:
            s = s.replace(token, "")
        cleaned_summaries.append(s.strip())
        
    # 제출 파일 생성
    submission = pd.DataFrame({'fname': fnames, 'summary': cleaned_summaries})
    
    result_dir = config['inference']['result_path']
    os.makedirs(result_dir, exist_ok=True)
    submission.to_csv(os.path.join(result_dir, 'submission.csv'), index=False, encoding='utf-8')
    
    print(f"Submission file saved to {os.path.join(result_dir, 'submission.csv')}")
    print("\n--- Sample Submission ---")
    print(submission.head())

if __name__ == "__main__":
    inference()