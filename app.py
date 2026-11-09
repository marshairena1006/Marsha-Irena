
import streamlit as st
import pandas as pd
import pickle
import joblib

# Load the trained model
try:
    # Mencoba me-load model.pkl yang berhasil dibuat sebelumnya
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception:
    model = None

# Streamlit app title
st.title('Prediksi Tagihan Listrik Jakarta')
st.write('Aplikasi untuk memprediksi jumlah tagihan listrik berdasarkan parameter yang diberikan.')

# Sidebar for user inputs
st.sidebar.header('Input Parameter')

def user_input_features():
    kwh = st.sidebar.slider('Konsumsi KWH (kWh)', 150.0, 600.0, 350.0)
    ac_units = st.sidebar.slider('Jumlah AC', 0, 3, 1)
    ac_hours_per_day = st.sidebar.slider('Jam AC per Hari', 0.0, 10.0, 5.0)
    family_size = st.sidebar.slider('Jumlah Anggota Keluarga', 2, 6, 4)
    
    month_name = st.sidebar.selectbox('Bulan', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    tariff_class = st.sidebar.selectbox('Kelas Tarif', ['R1', 'R2', 'R3'])
    
    data = {
        'kwh': kwh,
        'ac_units': ac_units,
        'ac_hours_per_day': ac_hours_per_day,
        'family_size': family_size,
        'month_name': month_name,
        'tariff_class': tariff_class
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

st.subheader('Parameter Input Pengguna:')
st.write(df_input)

# Define columns expected by model
training_columns_and_dtypes = {
    'kwh': 'float64',
    'ac_units': 'int64',
    'ac_hours_per_day': 'float64',
    'family_size': 'int64'
}

# Create DataFrame
final_input_df = pd.DataFrame(columns=training_columns_and_dtypes.keys())
for col, dtype in training_columns_and_dtypes.items():
    final_input_df[col] = final_input_df[col].astype(dtype)

# Populate numerical features
final_input_df.loc[0, 'kwh'] = df_input['kwh'][0]
final_input_df.loc[0, 'ac_units'] = df_input['ac_units'][0]
final_input_df.loc[0, 'ac_hours_per_day'] = df_input['ac_hours_per_day'][0]
final_input_df.loc[0, 'family_size'] = df_input['family_size'][0]

# Make prediction
if st.sidebar.button('Prediksi Tagihan'):
    if model is not None:
        try:
            prediction = model.predict(final_input_df)
            st.subheader('Hasil Prediksi Tagihan Listrik:')
            st.write(f"Tagihan Diprediksi: Rp {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
            st.exception(e)
    else:
        st.error("Model tidak berhasil dimuat. Pastikan file 'model.pkl' tersedia.")
