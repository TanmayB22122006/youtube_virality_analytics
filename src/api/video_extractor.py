import os
import pandas as pd
from youtube_client import get_service

def get_videos(yt, playlist_id, max_count=50):
    videos = []
    token = None
    
    while len(videos) < max_count:
        req = yt.playlistItems().list(
            playlistId=playlist_id,
            part='snippet',
            maxResults=50,
            pageToken=token
        )
        res = req.execute()
        
        for item in res.get('items', []):
            vid = item['snippet']['resourceId']['videoId']
            videos.append(vid)
            if len(videos) >= max_count:
                break
                
        token = res.get('nextPageToken')
        if not token:
            break
            
    return videos

def get_video_stats(yt, video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        req = yt.videos().list(
            id=','.join(batch),
            part='snippet,contentDetails,statistics'
        )
        res = req.execute()
        
        for item in res['items']:
            s = item.get('statistics', {})
            stats.append({
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'published_at': item['snippet']['publishedAt'],
                'duration': item['contentDetails']['duration'],
                'view_count': s.get('viewCount', 0),
                'like_count': s.get('likeCount', 0),
                'comment_count': s.get('commentCount', 0)
            })
    return stats

def get_comments(yt, video_id, max_count=50):
    comments = []
    try:
        req = yt.commentThreads().list(
            videoId=video_id,
            part='snippet',
            maxResults=max_count,
            textFormat='plainText'
        )
        res = req.execute()
        
        for item in res.get('items', []):
            c = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'comment_id': item['id'],
                'video_id': video_id,
                'text_display': c['textDisplay'],
                'author_name': c['authorDisplayName'],
                'like_count': c['likeCount']
            })
    except:
        pass
    return comments

def extract_all():
    yt = get_service()
    df_meta = pd.read_csv("data/raw/creators_metadata.csv")
    
    all_videos = []
    all_comments = []
    
    for _, row in df_meta.iterrows():
        p_id = row['uploads_playlist_id']
        vids = get_videos(yt, p_id, 50)
        
        v_stats = get_video_stats(yt, vids)
        for v in v_stats:
            v['creator_name'] = row['creator_name']
            all_videos.append(v)
            
        for vid in vids:
            c_data = get_comments(yt, vid, 50)
            all_comments.extend(c_data)
            
    df_v = pd.DataFrame(all_videos)
    df_c = pd.DataFrame(all_comments)
    
    df_v.to_csv("data/raw/raw_videos.csv", index=False)
    df_c.to_csv("data/raw/raw_comments.csv", index=False)

if __name__ == "__main__":
    extract_all()