# src/trainer.py

import os
import wandb
from rouge import Rouge
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, IntervalStrategy

def compute_metrics(eval_pred, tokenizer, remove_tokens):
    """ROUGE 점수를 계산하는 함수."""
    predictions, labels = eval_pred
    
    # 예측값과 실제값에서 패딩 토큰 ID를 실제 토큰으로 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

    # ROUGE 점수 계산을 위해 불필요한 토큰 제거
    for token in remove_tokens:
        decoded_preds = [pred.replace(token, "").strip() for pred in decoded_preds]
        decoded_labels = [label.replace(token, "").strip() for label in decoded_labels]

    # PDF 슬라이드 16: 한국어 특성을 고려한 형태소 단위(또는 어절) 평가
    # 여기서는 간단히 어절(띄어쓰기) 단위로 평가
    decoded_preds = [" ".join(pred.split()) for pred in decoded_preds]
    decoded_labels = [" ".join(label.split()) for label in decoded_labels]

    rouge = Rouge()
    scores = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    
    result = {key: value['f'] for key, value in scores.items()}
    
    # 로그를 위해 첫 3개의 예측/정답 쌍 출력
    print("\n--- Sample Predictions ---")
    for i in range(min(3, len(decoded_preds))):
        print(f"Pred: {decoded_preds[i]}")
        print(f"Gold: {decoded_labels[i]}\n")
        
    return result

def get_trainer(config: dict, model, tokenizer, train_dataset, val_dataset):
    """Seq2SeqTrainer를 설정하고 반환합니다."""
    
    # WandB 초기화
    if config['training']['report_to'] == 'wandb':
        wandb.init(
            entity=config['wandb']['entity'],
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            config=config
        )
        # 모델 체크포인트를 WandB에 저장하도록 설정
        os.environ["WANDB_LOG_MODEL"] = "true"
        os.environ["WANDB_WATCH"] = "false"

    training_args = Seq2SeqTrainingArguments(
        output_dir=config['output_dir'],
        logging_dir=config['logging_dir'],
        overwrite_output_dir=True,
        do_train=config['training']['do_train'],
        do_eval=config['training']['do_eval'],
        num_train_epochs=config['training']['num_train_epochs'],
        learning_rate=float(config['training']['learning_rate']),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        optim=config['training']['optim'],
        fp16=config['training']['fp16'],
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=config['training']['eval_steps'],
        save_strategy=IntervalStrategy.STEPS,
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        seed=config['seed'],
        predict_with_generate=config['training']['predict_with_generate'],
        generation_max_length=config['training']['generation_max_length'],
        report_to=config['training']['report_to'],
        metric_for_best_model="rouge-l" # ROUGE-L 점수를 기준으로 최고 성능 모델 저장
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    
    remove_tokens = config['model']['special_tokens'] + config['inference']['remove_tokens']
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, remove_tokens),
        callbacks=[early_stopping]
    )
    
    return trainer