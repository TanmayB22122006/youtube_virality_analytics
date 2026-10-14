import streamlit as st
import pandas as pd
from components import load_data, train_model, get_architecture_details, get_actionable_insights

st.set_page_config(page_title="YouTube Virality Engine", layout="wide")

# Fetch data and trained model from components
df_tgt, df_feat = load_data()
rf_model, encoders, feature_cols = train_model(df_tgt)

st.title("🚀 YouTube Virality Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📖 Model Architecture", "📊 Category-Wise Deep Insights"])

with tab1:
    st.header("Predict Video Virality")
    c1, c2 = st.columns(2)
    with c1:
        cat = st.selectbox("Category", encoders['category'].classes_)
        time_p = st.selectbox("Time Period", encoders['time_period'].classes_)
        day_w = st.selectbox("Day of Week", encoders['day_of_week'].classes_)
    with c2:
        st.markdown("**Video Duration**")
        col_h, col_m, col_s = st.columns(3)
        h = col_h.number_input("Hours", min_value=0, max_value=24, value=0)
        m = col_m.number_input("Minutes", min_value=0, max_value=59, value=12)
        s = col_s.number_input("Seconds", min_value=0, max_value=59, value=30)
        
        dur = (h * 3600) + (m * 60) + s
        if dur == 0: 
            dur = 10 
            
        rel_score = st.slider("Target Relatability Score (%)", 0.0, 100.0, 65.0, help="Set the expected emotional connection level of your script.")
        
    if st.button("Analyze & Predict"):
        input_data = pd.DataFrame([[cat, time_p, dur, rel_score, day_w]], columns=feature_cols)
        for col in ['category', 'time_period', 'day_of_week']:
            input_data[col] = encoders[col].transform(input_data[col])
            
        prob = rf_model.predict_proba(input_data)[0][1] * 100
        st.metric("Viral Probability Model Output", f"{prob:.2f}%")
        
        if prob >= 70:
            st.success("High potential for virality! The algorithm strongly favors these parameters.")
        elif prob >= 40:
            st.warning("Moderate chances. Consider tweaking the duration or focusing on audience relatability.")
        else:
            st.error("Low virality chances. Re-evaluate the upload time or length.")

with tab2:
    st.header("System Architecture & Engineering Details")
    st.markdown(get_architecture_details())

with tab3:
    st.header("Category-Wise Exploratory Data Analysis")
    
    c3, c4 = st.columns(2)
    cat_grp = df_tgt.groupby('category')[['engagement_rate', 'virality_score']].mean().reset_index()
    time_grp = df_tgt.groupby('time_period')['virality_score'].mean().reset_index()
    
    with c3:
        st.markdown("**Avg Virality Score by Category**")
        st.bar_chart(cat_grp.set_index('category')['virality_score'])
        
    with c4:
        st.markdown("**Virality Trends across Time Periods**")
        st.area_chart(time_grp.set_index('time_period')['virality_score'])
        
    st.markdown("**Relatability vs Engagement Distribution**")
    st.scatter_chart(data=df_tgt, x='relatability_score', y='engagement_rate', color='category')

    st.divider()
    st.subheader("Actionable Insights for Content Creators")
    st.markdown(get_actionable_insights())