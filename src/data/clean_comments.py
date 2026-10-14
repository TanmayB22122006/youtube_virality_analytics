import pandas as pd
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_comment_data():
    df = pd.read_csv("data/raw/raw_comments.csv")
    df = df.dropna(subset=['text_display'])
    
    df['clean_text'] = df['text_display'].apply(clean_text)
    df = df[df['clean_text'] != '']
    
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    
    df.to_csv("data/processed/cleaned_comments.csv", index=False)

if __name__ == "__main__":
    clean_comment_data()