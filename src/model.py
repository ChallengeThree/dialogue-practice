# src/model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(config: dict):
    """설정 파일에 명시된 모델 타입에 따라 모델과 토크나이저를 동적으로 불러옵니다."""
    model_type = config['model']['type']
    model_name = config['model']['architectures'][model_type]['name']
    
    # AutoClass를 사용하여 모델 타입에 맞는 클래스를 자동으로 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 스페셜 토큰 추가
    special_tokens = config['model']['special_tokens']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"Model type '{model_type}' with name '{model_name}' loaded.")
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer