import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# --- Memuat model dan transformer ---
try:
    with open('catboost_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('power_transformer.pkl', 'rb') as file:
        pt = pickle.load(file)
    with open('standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("File model atau transformer tidak ditemukan. Pastikan semua file .pkl berada dalam folder yang sama dengan app.py.")
    st.stop()

# --- Mapping kategori ---
fuel_mapping = {
    'Diesel': 0,
    'Petrol': 1,
    'CNG': 2,
    'LPG': 3,
    'Electric': 4
}
seller_type_mapping = {
    'Individual': 0,
    'Dealer': 1,
    'Trustmark Dealer': 2
}
transmission_mapping = {
    'Manual': 0,
    'Automatic': 1
}
owner_mapping = {
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
}

# --- Tampilan Aplikasi ---
st.title('Prediksi Harga Mobil Bekas')
st.write('Aplikasi ini memprediksi harga jual mobil bekas berdasarkan beberapa parameter.')

st.sidebar.header('Input Detail Mobil')

# --- Input tahun & km ---
year_input = st.sidebar.number_input('Tahun Mobil', min_value=1992, max_value=2020, value=2015, step=1)
km_driven_input = st.sidebar.number_input('Jarak Tempuh (km)', min_value=0, max_value=1000000, value=50000, step=1000)

# --- Input kategori ---
fuel_type_input = st.sidebar.selectbox('Jenis Bahan Bakar', list(fuel_mapping.keys()))
seller_type_input = st.sidebar.selectbox('Tipe Penjual', list(seller_type_mapping.keys()))
transmission_type_input = st.sidebar.selectbox('Transmisi', list(transmission_mapping.keys()))
owner_status_input = st.sidebar.selectbox('Jumlah Pemilik Sebelumnya', list(owner_mapping.keys()))

# --- Autocomplete nama mobil ---
try:
    valid_car_names_df = pd.read_csv('X_train_names.csv')
    valid_car_names = valid_car_names_df['name'].unique().tolist()
except FileNotFoundError:
    st.error("File X_train_names.csv tidak ditemukan. Pastikan sudah ada.")
    st.stop()

car_name_input = st.sidebar.selectbox(
    'Nama / Merek Mobil',
    options=valid_car_names
)

# --- Fitur Pilihan Kurs ---
st.sidebar.subheader("Konversi Mata Uang INR → IDR")

kurs_option = st.sidebar.selectbox(
    "Pilih Sumber Kurs",
    ["Kurs Otomatis (API)", "Kurs Manual"]
)

INR_TO_IDR = 190  # kurs default

if kurs_option == "Kurs Otomatis (API)":
    try:
        url = "https://open.er-api.com/v6/latest/INR"
        data = requests.get(url).json()
        INR_TO_IDR = data["rates"]["IDR"]
        st.sidebar.success(f"Kurs otomatis dimuat: 1 INR = Rp {INR_TO_IDR:,.2f}")
    except:
        st.sidebar.warning("Gagal mengambil kurs otomatis. Menggunakan kurs default (190).")
else:
    INR_TO_IDR = st.sidebar.number_input(
        "Masukkan kurs INR → IDR",
        min_value=1.0, max_value=2000.0,
        value=190.0, step=1.0
    )

# --- Prediksi ---
if st.sidebar.button('Prediksi Harga Mobil'):

    # Encode kategori
    fuel_encoded = fuel_mapping[fuel_type_input]
    seller_type_encoded = seller_type_mapping[seller_type_input]
    transmission_encoded = transmission_mapping[transmission_type_input]
    owner_encoded = owner_mapping[owner_status_input]

    # PowerTransformer hanya untuk km_driven
    data_for_pt = pd.DataFrame([[0.0, km_driven_input]], columns=['selling_price', 'km_driven'])
    transformed_data_for_pt = pt.transform(data_for_pt)
    km_driven_yj = transformed_data_for_pt[0, 1]

    # StandardScaler untuk km_driven_yj
    data_for_scaler = pd.DataFrame([[km_driven_yj]], columns=['km_driven_yj'])
    scaled_km = scaler.transform(data_for_scaler)[0][0]

    # Susun input DF
    prediction_df = pd.DataFrame([[
        fuel_encoded,
        seller_type_encoded,
        transmission_encoded,
        owner_encoded,
        scaled_km,
        year_input,
        car_name_input
    ]], columns=[
        'fuel', 'seller_type', 'transmission', 'owner',
        'km_driven_yj', 'year', 'name'
    ])

    # Prediksi di skala YJ
    predicted_price_yj = model.predict(prediction_df)[0]

    # Kembalikan ke skala asli INR
    data_for_inverse_pt = pd.DataFrame([[predicted_price_yj, 0.0]], columns=['selling_price', 'km_driven'])
    original_scale_prediction = pt.inverse_transform(data_for_inverse_pt)
    final_price_inr = original_scale_prediction[0, 0]

    # Konversi ke IDR
    final_price_idr = final_price_inr * INR_TO_IDR

    # --- Output ---
    st.subheader('Detail Input Anda')
    st.write(pd.DataFrame([{
        'Tahun': year_input,
        'Jarak Tempuh (km)': km_driven_input,
        'Jenis Bahan Bakar': fuel_type_input,
        'Tipe Penjual': seller_type_input,
        'Transmisi': transmission_type_input,
        'Jumlah Pemilik': owner_status_input,
        'Nama / Merek Mobil': car_name_input
    }]))

    st.subheader('Hasil Prediksi Harga')
    st.success(f"Harga Estimasi (INR): ₹ {final_price_inr:,.2f}")
    st.success(f"Harga Estimasi (IDR): Rp {final_price_idr:,.0f}")
    st.caption(f"Kurs yang digunakan: 1 INR = Rp {INR_TO_IDR:,.2f}")

    st.markdown("""
    **Catatan:**
    * Prediksi ini merupakan estimasi.
    * Faktor kondisi mobil, lokasi, dan fitur tambahan dapat mempengaruhi harga sebenarnya.
    * Model dilatih berdasarkan data dari CarDekho (India).
    """)
