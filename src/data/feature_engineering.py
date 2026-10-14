import pandas as pd

def get_length_cat(seconds):
    if seconds <= 60:
        return 'Shorts'
    elif seconds <= 600:
        return 'Medium'
    else:
        return 'Long'

def get_time_period(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

def create_features():
    df = pd.read_csv("data/processed/cleaned_videos.csv")
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    df['length_category'] = df['duration_seconds'].apply(get_length_cat)
    df['day_of_week'] = df['published_at'].dt.day_name()
    df['time_period'] = df['published_at'].dt.hour.apply(get_time_period)
    
    v = df['view_count'].replace(0, 1)
    l = df['like_count']
    c = df['comment_count']
    
    df['engagement_rate'] = ((l + c) / v) * 100
    df['virality_score'] = ((l * 1.5 + c * 3) / v) * 100
    
    df.to_csv("data/processed/engineered_features.csv", index=False)

if __name__ == "__main__":
    create_features()