# inference_gpt4.py

import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from src.utils import load_config

def summarize_with_gpt4(client, dialogue: str, model_name: str) -> str:
    """GPT-4 API를 사용하여 대화를 요약합니다."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes dialogues concisely and accurately in Korean."},
                {"role": "user", "content": f"Please summarize the following dialogue:\n\n{dialogue}"}
            ],
            temperature=0.5,
            max_tokens=150,
        )
        summary = response.choices[0].message.content
        return summary.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error in summarization."

def main():
    config = load_config("config/config.yaml")
    
    api_key = config['gpt4'].get('api_key')
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        print("OpenAI API key is not set in config/config.yaml. Please set it to run GPT-4 inference.")
        return
        
    client = OpenAI(api_key=api_key)
    model_name = config['gpt4']['model_name']
    
    # 테스트 데이터 로드
    test_data_path = os.path.join(config['data_dir'], 'test.csv')
    df = pd.read_csv(test_data_path)
    
    summaries = []
    print(f"Starting summarization with {model_name}...")
    for dialogue in tqdm(df['dialogue'], desc="GPT-4 Summarization"):
        summary = summarize_with_gpt4(client, dialogue, model_name)
        summaries.append(summary)
        
    # 결과 저장
    submission = pd.DataFrame({'fname': df['fname'], 'summary': summaries})
    result_dir = config['inference']['result_path']
    os.makedirs(result_dir, exist_ok=True)
    submission_path = os.path.join(result_dir, 'submission_gpt4.csv')
    submission.to_csv(submission_path, index=False, encoding='utf-8')
    
    print(f"\nGPT-4 submission file saved to {submission_path}")
    print("\n--- Sample GPT-4 Submission ---")
    print(submission.head())

if __name__ == "__main__":
    main()