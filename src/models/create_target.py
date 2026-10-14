import pandas as pd
import numpy as np

def create_target():
    df_v = pd.read_csv("data/processed/engineered_features.csv")
    df_c = pd.read_csv("data/processed/nlp_analyzed_comments.csv")
    
    df = pd.merge(df_v, df_c, on='video_id', how='inner')
    
    cat_map = {
        "Physics Wallah": "EdTech", "Prashant Kirad": "EdTech", "Aman Dhattarwal": "EdTech", "Khan GS Research Centre": "EdTech", "Dear Sir": "EdTech",
        "Raj Shamani": "Finance", "Ankur Warikoo": "Finance", "Akshat Shrivastava": "Finance", "Pranjal Kamra": "Finance", "CA Rachana Ranade": "Finance",
        "Nitish Rajput": "Infotainment", "Professor Of How": "Infotainment", "Dhruv Rathee": "Infotainment", "Abhi and Niyu": "Infotainment", "Mohak Mangal": "Infotainment",
        "Samay Raina": "Entertainment", "Total Gaming": "Entertainment", "Khooni Monday": "Entertainment", "Mr Indian Hacker": "Entertainment", "CarryMinati": "Entertainment",
        "Sourav Joshi Vlogs": "Vlogs", "Flying Beast": "Vlogs", "Mumbiker Nikhil": "Vlogs", "Gaurav Chaudhary": "Vlogs", "Wanderers Hub": "Vlogs"
    }
    
    if 'category' not in df.columns and 'creator_name' in df.columns:
        df['category'] = df['creator_name'].map(cat_map)
        
    threshold = df['virality_score'].quantile(0.75)
    df['is_viral'] = (df['virality_score'] >= threshold).astype(int)
    
    df.to_csv("data/processed/virality_target_data.csv", index=False)

if __name__ == "__main__":
    create_target()