# train.py

import os
import torch
import wandb
from datetime import datetime
from src.utils import load_config
from src.data_loader import DataPreprocessor
from src.model import load_model_and_tokenizer
from src.trainer import get_trainer
from create_dummy_data import create_data_file

def main():
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # WandB 실행 이름 자동 생성
    if not config['wandb']['name']:
        model_type = config['model']['type']
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        config['wandb']['name'] = f"{model_type}-{timestamp}"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 테스트 모드일 경우 더미 데이터 생성
    if config.get('test_mode', False):
        print("Test mode enabled. Generating dummy data...")
        num_train_samples = 100
        num_val_samples = 20
        create_data_file('train_test.csv', num_train_samples)
        create_data_file('val_test.csv', num_val_samples)
        print("Dummy data generation complete.")

    model, tokenizer = load_model_and_tokenizer(config)
    model.to(device)

    preprocessor = DataPreprocessor(config, tokenizer)
    train_dataset, val_dataset = preprocessor.setup_datasets()

    # 데이터셋 크기 축소 (테스트용)
    if config.get('test_mode', False):
        # SummarizationDataset은 select 메서드가 없으므로 직접 슬라이싱
        train_dataset.data = {k: v[:min(len(train_dataset), 50)] for k, v in train_dataset.data.items()}
        train_dataset.data_len = len(train_dataset.data['input_ids'])
        val_dataset.data = {k: v[:min(len(val_dataset), 10)] for k, v in val_dataset.data.items()}
        val_dataset.data_len = len(val_dataset.data['input_ids'])
    else:
        # 실제 데이터셋의 경우에도 슬라이싱으로 변경
        train_dataset.data = {k: v[:min(len(train_dataset), 1000)] for k, v in train_dataset.data.items()}
        train_dataset.data_len = len(train_dataset.data['input_ids'])
        val_dataset.data = {k: v[:min(len(val_dataset), 1000)] for k, v in val_dataset.data.items()}
    
    print(f"Train dataset size (sampled): {len(train_dataset)}")
    print(f"Validation dataset size (sampled): {len(val_dataset)}")

    trainer = get_trainer(config, model, tokenizer, train_dataset, val_dataset)

    print("Training started...")
    trainer.train()
    print("Training finished.")

    best_model_path = os.path.join(config['output_dir'], "best_model")
    trainer.save_model(best_model_path)
    # 토크나이저도 함께 저장해야 추론 시 완벽하게 불러올 수 있습니다.
    tokenizer.save_pretrained(best_model_path)
    print(f"Best model and tokenizer saved to {best_model_path}")
    
    if config['training']['report_to'] == 'wandb':
        wandb.finish()

if __name__ == "__main__":
    main()