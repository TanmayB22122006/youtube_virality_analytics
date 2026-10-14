import pandas as pd

def generate_strategy(creator_name=None):
    df = pd.read_csv("data/processed/virality_target_data.csv")
    
    if creator_name:
        df = df[df['creator_name'] == creator_name]
        
    viral_df = df[df['is_viral'] == 1]
    
    if viral_df.empty:
        viral_df = df.sort_values(by='virality_score', ascending=False).head(5)
        
    best_len = viral_df['length_category'].mode()[0]
    best_time = viral_df['time_period'].mode()[0]
    best_day = viral_df['day_of_week'].mode()[0]
    avg_rel = viral_df['relatability_score'].mean()
    
    strategy = {
        "creator": creator_name if creator_name else "Overall",
        "recommended_length": best_len,
        "best_upload_time": best_time,
        "best_upload_day": best_day,
        "target_relatability_score": round(avg_rel, 2)
    }
    
    return strategy

if __name__ == "__main__":
    s = generate_strategy()
    for k, v in s.items():
        print(f"{k}: {v}")