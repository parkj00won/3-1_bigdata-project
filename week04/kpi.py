import streamlit as st 
# 화면을 3등분하여 각 열에 metric 배치 
col1, col2, col3 = st.columns(3) 
col1.metric("온도", "23°C", "1.5°C") 
col2.metric("습도", "45%", "-2%") 
col3.metric("풍속", "12km/h", "3km/h") 