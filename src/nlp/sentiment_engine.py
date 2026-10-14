import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

def check_relatable(text, subj):
    text = str(text).lower()
    pronouns = r'\b(i|me|my|mine|we|us|our|ours)\b'
    has_pro = bool(re.search(pronouns, text))
    
    if subj >= 0.4 and has_pro:
        return 1
    return 0

def run_nlp():
    df = pd.read_csv("data/processed/cleaned_comments.csv")
    df = df.dropna(subset=['clean_text'])
    
    analyzer = SentimentIntensityAnalyzer()
    
    pol = []
    subj_list = []
    flags = []
    
    for text in df['clean_text']:
        text_str = str(text)
        
        vs = analyzer.polarity_scores(text_str)
        pol.append(vs['compound'])
        
        tb = TextBlob(text_str)
        s = tb.sentiment.subjectivity
        subj_list.append(s)
        
        flags.append(check_relatable(text_str, s))
        
    df['polarity'] = pol
    df['subjectivity'] = subj_list
    df['is_relatable'] = flags
    
    v_df = df.groupby('video_id').agg(
        total_c=('video_id', 'count'),
        relatable_c=('is_relatable', 'sum')
    ).reset_index()
    
    v_df['relatability_score'] = (v_df['relatable_c'] / v_df['total_c']) * 100
    v_df['relatability_score'] = v_df['relatability_score'].round(2)
    
    v_df.to_csv("data/processed/nlp_analyzed_comments.csv", index=False)

if __name__ == "__main__":
    run_nlp()