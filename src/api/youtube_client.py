import os
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

def get_service():
    api_key = os.getenv("YOUTUBE_API_KEY")
    return build('youtube', 'v3', developerKey=api_key)

def fetch_creator_metadata(creators_list):
    yt = get_service()
    data = []
    
    for name in creators_list:
        req = yt.search().list(q=name, type='channel', part='id,snippet', maxResults=1)
        res = req.execute()
        
        if res.get('items'):
            item = res['items'][0]
            ch_id = item['id']['channelId']
            
            ch_req = yt.channels().list(id=ch_id, part='contentDetails')
            ch_res = ch_req.execute()
            
            if ch_res.get('items'):
                uploads_id = ch_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                data.append({
                    'creator_name': name,
                    'channel_id': ch_id,
                    'uploads_playlist_id': uploads_id
                })
                
    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/creators_metadata.csv", index=False)
    return df

if __name__ == "__main__":
    creators = [
        "Physics Wallah", "Prashant Kirad", "Aman Dhattarwal", "Khan GS Research Centre", "Dear Sir",
        "Raj Shamani", "Ankur Warikoo", "Akshat Shrivastava", "Pranjal Kamra", "CA Rachana Ranade",
        "Nitish Rajput", "Professor Of How", "Dhruv Rathee", "Abhi and Niyu", "Mohak Mangal",
        "Samay Raina", "Total Gaming", "Khooni Monday", "Mr Indian Hacker", "CarryMinati",
        "Sourav Joshi Vlogs", "Flying Beast", "Mumbiker Nikhil", "Gaurav Chaudhary", "Wanderers Hub"
    ]
    fetch_creator_metadata(creators)