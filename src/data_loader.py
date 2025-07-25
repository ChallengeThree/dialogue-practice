# src/data_loader.py

import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset


class SummarizationDataset(Dataset):
    def __init__(self, tokenized_data, data_len):
        self.data = tokenized_data
        self.data_len = data_len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.data.items()}
        return item

    def __len__(self):
        return self.data_len

class DataPreprocessor:
    def __init__(self, config: dict, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.bos_token = self.tokenizer.bos_token or ""
        self.eos_token = self.tokenizer.eos_token or ""
        
        # 모델별 접두사 설정
        model_type = self.config['model']['type']
        self.prefix = self.config['model']['architectures'][model_type]['prefix']

    def clean_text(self, text: str) -> str:
        # 1. 괄호 안 내용 제거 (e.g., (웃음), [참고])
        text = re.sub(r'\([^)]*\)|\[[^\]]*\]', '', text)
        
        # 2. 번역투 표현 완화
        # 어색한 대명사 제거 (문장 시작 또는 공백 뒤에 오는 경우)
        text = re.sub(r'(^|\s)(그는|그녀는|당신은)\s', ' ', text)
        # 기타 번역투 표현 변경
        text = text.replace("에 대하여", "에 대해")
        text = text.replace("을 가지고 있다", "이 있다")
        text = text.replace("를 가지고 있다", "가 있다")

        # 3. 구두점 주변 공백 정규화
        # 구두점 앞의 공백 제거
        text = re.sub(r'\s+([.,?!%])', r'\1', text)
        # 구두점 뒤에 공백 추가 (이미 공백이 있는 경우는 유지)
        text = re.sub(r'([.,?!%])(?=[가-힣A-Za-z0-9])', r'\1 ', text)

        # 4. 한글, 영어, 숫자, 및 특정 특수문자(#, \n, ., ,, ?, !, %)를 제외한 나머지 제거
        text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9#\n.,?!% ]', '', text)
        
        # 5. 반복되는 자음, 모음, 특수문자 처리 (e.g., ㅋㅋㅋ -> ㅋ, !!! -> !)
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ.,?!])\1{2,}', r'\1', text)
        
        # 6. 여러 개의 공백을 하나로 줄이고 양쪽 끝의 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def prepare_data(self, data_path: str):
        print(f"Starting to load data from {data_path}...")
        df = pd.read_csv(data_path, index_col=0)
        print(f"Finished loading data from {data_path}. Shape: {df.shape}")

        # 데이터 타입 강제 및 결측치 처리
        df = df.astype({'dialogue': str, 'summary': str})
        df.dropna(subset=['dialogue', 'summary'], inplace=True)

        # prefix 추가 로직
        df['dialogue'] = self.prefix + df['dialogue'].apply(self.clean_text)

        if 'summary' in df.columns:
            df['summary'] = df['summary'].apply(self.clean_text)
            encoder_input = df['dialogue'].tolist()
            decoder_input = [self.bos_token + s for s in df['summary'].tolist()]
            decoder_output = [s + self.eos_token for s in df['summary'].tolist()]
            return encoder_input, decoder_input, decoder_output
        else: # For test set
            encoder_input = df['dialogue'].tolist()
            return encoder_input, df['fname'].tolist()


    def tokenize_data(self, encoder_input, decoder_input=None, decoder_output=None):
        print("Tokenizing encoder input...")
        tokenized_encoder = self.tokenizer(
            encoder_input,
            max_length=self.config['model']['encoder_max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        print("Encoder input tokenization complete.")

        if decoder_input and decoder_output:
            print("Tokenizing decoder input and output...")
            tokenized_decoder_input = self.tokenizer(
                decoder_input,
                max_length=self.config['model']['decoder_max_len'],
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            tokenized_decoder_output = self.tokenizer(
                decoder_output,
                max_length=self.config['model']['decoder_max_len'],
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            print("Decoder input and output tokenization complete.")
            
            labels = tokenized_decoder_output['input_ids']
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            print("Finished tokenizing data.")

            return {
                'input_ids': tokenized_encoder.input_ids,
                'attention_mask': tokenized_encoder.attention_mask,
                'decoder_input_ids': tokenized_decoder_input.input_ids,
                'decoder_attention_mask': tokenized_decoder_input.attention_mask,
                'labels': labels
            }
        
        return {
            'input_ids': tokenized_encoder.input_ids,
            'attention_mask': tokenized_encoder.attention_mask
        }

    def setup_datasets(self):
        print("Starting setup_datasets...")
        if self.config.get('test_mode', False):    
            train_path = os.path.join(self.config['data_dir'], 'train_test.csv')
            val_path = os.path.join(self.config['data_dir'], 'val_test.csv')
        else:
            train_path = os.path.join(self.config['data_dir'], 'train.csv')
            val_path = os.path.join(self.config['data_dir'], 'val.csv')

        enc_train, dec_in_train, dec_out_train = self.prepare_data(train_path)
        enc_val, dec_in_val, dec_out_val = self.prepare_data(val_path)

        tokenized_train_cache_path = os.path.join(self.config['data_dir'], f"cached_tokenized_train{'_test' if self.config.get('test_mode', False) else ''}.pt")
        tokenized_val_cache_path = os.path.join(self.config['data_dir'], f"cached_tokenized_val{'_test' if self.config.get('test_mode', False) else ''}.pt")

        if os.path.exists(tokenized_train_cache_path) and os.path.exists(tokenized_val_cache_path):
            print("Loading tokenized data from cache...")
            tokenized_train = torch.load(tokenized_train_cache_path)
            tokenized_val = torch.load(tokenized_val_cache_path)
        else:
            print("Tokenizing data and saving to cache...")
            tokenized_train = self.tokenize_data(enc_train, dec_in_train, dec_out_train)
            tokenized_val = self.tokenize_data(enc_val, dec_in_val, dec_out_val)
            torch.save(tokenized_train, tokenized_train_cache_path)
            print(f"Train tokenized data saved to {tokenized_train_cache_path}")
            torch.save(tokenized_val, tokenized_val_cache_path)
            print(f"Validation tokenized data saved to {tokenized_val_cache_path}")

        train_dataset = SummarizationDataset(tokenized_train, len(enc_train))
        val_dataset = SummarizationDataset(tokenized_val, len(enc_val))

        print("Finished setup_datasets.")
        return train_dataset, val_dataset