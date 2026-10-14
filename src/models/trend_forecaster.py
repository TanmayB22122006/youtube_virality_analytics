import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def forecast_trends():
    df = pd.read_csv("data/processed/engineered_features.csv")
    
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    df['date'] = df['published_at'].dt.date
    
    daily_df = df.groupby('date')['engagement_rate'].mean().reset_index()
    daily_df = daily_df.sort_values('date')
    
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    start_d = daily_df['date'].min()
    daily_df['days_passed'] = (daily_df['date'] - start_d).dt.days
    
    X = daily_df[['days_passed']]
    y = daily_df['engagement_rate']
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    last_d = daily_df['days_passed'].max()
    X_fut = pd.DataFrame({'days_passed': [last_d + i for i in range(1, 15)]})
    y_fut = lr.predict(X_fut)
    
    dates_fut = [daily_df['date'].max() + timedelta(days=i) for i in range(1, 15)]
    res_df = pd.DataFrame({
        'forecast_date': [d.date() for d in dates_fut],
        'expected_engagement_rate': np.maximum(0, y_fut).round(2)
    })
    
    print(res_df.head(7).to_string(index=False))
    res_df.to_csv("data/processed/trend_forecast.csv", index=False)

if __name__ == "__main__":
    forecast_trends()