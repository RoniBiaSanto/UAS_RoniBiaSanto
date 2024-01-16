import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model
lr = pickle.load(open('ronibiasanto1.pkl','rb'))

#load dataset
data = pd.read_csv('Breast_Cancern.csv')
data = data[['diagnosis','radius_mean','area_mean', 'radius_se', 'area_se', 'smoothness_mean','smoothness_se']]
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)

st.title('Aplikasi Sederhana Deteksi Gejala Kanker Payudara').\
    markdown("<p style='color: #ff5733; font-weight: bold; font-size: 24px; text-align: center;'>Aplikasi Sederhana Deteksi Gejala Kanker Payudara</p>", unsafe_allow_html=True)

html_layout1 = """
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @keyframes rollText {
            0% {
                transform: translateX(100%);
            }
            100% {
                transform: translateX(-100%);
            }
        }

        .custom-container {
            background-color: #3498db;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            text-align: center;
        }

        .custom-container h1 {
            color: white;
            font-size: 48px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; /* Menghapus margin bawaan dari <h1> */
            white-space: nowrap; /* Mencegah teks pindah ke baris baru */
            overflow: hidden; /* Menyembunyikan teks yang keluar dari kotak */
        }

        .animated-text {
            display: inline-block;
            animation: rollText 4s linear infinite;
        }

        .custom-container p {
            color: white;
            font-size: 20px;
            font-family: 'Arial', sans-serif;
        }

        .custom-container p.small-text {
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="custom-container">
        <h1><span class="animated-text">Deteksi Dini Kanker Payudara</span></h1>
        <br>
        <p>Jangan biarkan penyakit menghentikan kebahagiaan Anda.<br>Deteksi kanker payudara lebih cepat, hidup lebih lama!</p>
        <br>
        <p class="small-text">Gunakan aplikasi ini untuk pemeriksaan mandiri secara berkala.<br>Kesehatan Anda adalah prioritas utama.</p>
    </div>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Logistic Regression']
option = st.sidebar.selectbox('Model Algoritma',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Set Breast Cancer (Diagnostik)</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #333;
            background-color: #f8f8f8;
            margin: 30px;
        }
        h1 {
            color: #4d4dff;
            text-align: center;
        }
        p {
            font-size: 1.2em;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Data Set Breast Cancer (Diagnostik)</h1>
    <p>Ini adalah kumpulan data mengenai kanker payudara.</p>
</body>

    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('ticks')

if st.checkbox('EDA'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
x = data.drop('diagnosis',axis=1)
y = data['diagnosis']
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
  st.subheader('x_train')
  st.write(x_train.head())
  st.write(x_train.shape)
  st.subheader("y_train")
  st.write(y_train.head())
  st.write(y_train.shape)
  st.subheader('x_test')
  st.write(x_test.shape)
  st.subheader('y_test')
  st.write(y_test.head())
  st.write(y_test.shape)

def user_report():
  radius_mean = st.sidebar.slider('radius_mean: ', 0.0, 28.11000, 21.0)
  area_mean = st.sidebar.slider('area_mean: ', 0.0, 2501.0, 21.0)
  radius_se = st.sidebar.slider('radius_se: ', 0.0, 2.87300, 1.5)
  area_se = st.sidebar.slider('area_se: ', 0.0, 542.20, 200.0)
  smoothness_mean = st.sidebar.slider('smoothness_mean: ', 0.0, 0.16340, 0.12)
  smoothness_se = st.sidebar.slider('smoothness_se: ', 0.0, 0.03113, 0.021)

  user_report_data = {
    'radius_mean':radius_mean,
    'area_mean':area_mean,
    'radius_se':radius_se,
    'area_se':area_se,
    'smoothness_mean':smoothness_mean,
    'smoothness_se':smoothness_se,
  }
  report_data = pd.DataFrame(user_report_data,index=[0])
  return report_data

# Data Pasien
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

# Prediksi dengan penanganan kesalahan
try:
    if lr is not None:  # Periksa apakah model sudah diinisialisasi
        user_result = lr.predict(user_data)
        lr_score = accuracy_score(y_test, lr.predict(x_test))
    else:
        st.error("Model belum  dilatih.")
        user_result = None
        lr_score = None
except Exception as e:
    st.error(f"Error predicting user data: {e}")
    user_result = None
    lr_score = None

# Tampilkan hasil prediksi dan skor model
if user_result is not None:
    st.success("Prediksi berhasil:")
    st.write(user_result)
    st.write(f"Akurasi Model: {lr_score}")
else:
    st.warning("Prediksi tidak dapat dilakukan karena terjadi kesalahan.")
