import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Forecasting Permintaan (ARIMA)")

# === LOAD MODEL YANG SUDAH DIFIT ===
with open("forecasting_permintaan.sav","rb") as f:
    model = pickle.load(f)

# pilih jumlah langkah ke depan
year = int(st.slider("Jumlah periode yang diprediksi", 1, 30, 1))

if st.button("Prediksi"):

    # multi-step forecast
    fc = model.get_forecast(steps=year)

    # nilai prediksi
    pred = fc.predicted_mean

    # rapikan tabel
    pred_df = pd.DataFrame({
        "Periode ke-": range(1, year+1),
        "Forecast PO": pred.values
    })

    # tampilkan ke layar
    st.subheader("Hasil Forecast")
    st.dataframe(pred_df)

    # grafik
    fig, ax = plt.subplots()
    ax.plot(pred_df["Periode ke-"], pred_df["Forecast PO"])
    ax.set_xlabel("Periode ke-")
    ax.set_ylabel("Nilai Ramalan")
    ax.set_title("Forecast ARIMA Multi-step")
    st.pyplot(fig)

    st.success(f"Total titik prediksi: {len(pred_df)}")
