import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', sorted(df['Company'].unique()))
type_name = st.selectbox('Type', sorted(df['TypeName'].unique()))
ram = st.selectbox('RAM (in GB)', sorted(df['Ram(GB)'].unique()))

weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=10.0, value=2.0)

touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 15.6)

resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
     '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)

cpu = st.selectbox('CPU Brand', sorted(df['Cpu Brand'].unique()))

ssd = st.selectbox('SSD (in GB)', sorted(df['SSD'].unique()))
hdd = st.selectbox('HDD (in GB)', sorted(df['HDD'].unique()))

gpu = st.selectbox('GPU Brand', sorted(df['GPU Brand'].unique()))
os_name = st.selectbox('OS', sorted(df['os'].unique()))

if st.button('Predict Price'):
    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = (((X_res ** 2) + (Y_res ** 2)) ** 0.5) / screen_size

    query_df = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram(GB)': ram,
        'Weight(kg)': weight,
        'TouchScreen': touchscreen,   # keep Yes/No
        'IPS': ips,                   # keep Yes/No
        'Pixel_Per_Inch': ppi,
        'Cpu Brand': cpu,
        'SSD': ssd,
        'HDD': hdd,
        'GPU Brand': gpu,
        'os': os_name
    }])

    pred = pipe.predict(query_df)[0]

    # keep exp only if training was log(price)
    predicted_price = int(np.exp(pred))

    st.success(f"The predicted price of this configuration is ₹ {predicted_price}")
