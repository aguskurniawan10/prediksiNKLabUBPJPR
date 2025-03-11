#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Path untuk model dan preprocessing tools
MODEL_PATH = "best_model.pkl"
IMPUTER_PATH = "imputer.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH = Path(r"C:\Users\agus.kurniawan\Downloads\DATA PREDIKSI NK LAB 2025.xlsx")  # Sesuaikan path ini

def train_and_save_model():
    """Melatih model dan menyimpannya menggunakan pickle."""
    
    # Load Data
    df = pd.read_excel(DATA_PATH)

    # Standarisasi nama kolom (hapus spasi)
    df.columns = df.columns.str.strip()

    # Pastikan kolom yang diperlukan ada dalam dataset
    required_columns = ['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
                        'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING', 'GCV (ARB) LAB']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam dataset!")

    # Encode Suppliers
    label_encoder = LabelEncoder()
    df['Suppliers'] = label_encoder.fit_transform(df['Suppliers'])

    # Fitur dan target
    X = df[['Suppliers', 'GCV ARB UNLOADING', 'TM ARB UNLOADING', 
            'Ash Content ARB UNLOADING', 'Total Sulphur ARB UNLOADING']]
    y = df['GCV (ARB) LAB']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pisahkan fitur numerik (tanpa 'Suppliers')
    X_train_numeric = X_train.drop(columns=['Suppliers'])
    X_test_numeric = X_test.drop(columns=['Suppliers'])

    # Imputasi data numerik
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_numeric)
    X_test_imputed = imputer.transform(X_test_numeric)

    # Normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Gabungkan kembali dengan Suppliers
    X_train_final = np.hstack([X_train[['Suppliers']].values, X_train_scaled])
    X_test_final = np.hstack([X_test[['Suppliers']].values, X_test_scaled])

    # Model yang akan diuji
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf')
    }

    # Latih dan evaluasi model
    best_model = None
    best_score = float('-inf')
    results = {}

    for name, model in models.items():
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        r2 = r2_score(y_test, y_pred)
        results[name] = r2
        if r2 > best_score:
            best_score = r2
            best_model = model

    best_model_name = max(results, key=results.get)
    print(f"Model terbaik: {best_model_name} dengan R2: {best_score:.4f}")

    # Simpan model dan preprocessing tools
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    with open(IMPUTER_PATH, "wb") as file:
        pickle.dump(imputer, file)

    with open(SCALER_PATH, "wb") as file:
        pickle.dump(scaler, file)

    with open(ENCODER_PATH, "wb") as file:
        pickle.dump((label_encoder, label_encoder.classes_), file)

    print("Model dan preprocessing tools berhasil disimpan!")


# **Cek apakah model sudah ada, jika belum, latih dan simpan model**
if not os.path.exists(MODEL_PATH):
    st.write("Melatih model, harap tunggu...")
    train_and_save_model()

# **Load model dan preprocessing tools**
with open(MODEL_PATH, "rb") as file:
    best_model = pickle.load(file)

with open(IMPUTER_PATH, "rb") as file:
    imputer = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

with open(ENCODER_PATH, "rb") as file:
    label_encoder, classes = pickle.load(file)
    label_encoder.classes_ = classes  # Pastikan encoding tetap sama

# **Streamlit UI**
st.title("Prediksi GCV (ARB) LAB")
st.write("Masukkan nilai parameter untuk mendapatkan prediksi.")

# **Input fields**
supplier_options = list(label_encoder.classes_)
supplier_selected = st.selectbox("Suppliers", supplier_options)
supplier_encoded = label_encoder.transform([supplier_selected])[0]

gcv_arb_unloading = st.number_input("GCV ARB UNLOADING", value=4200.0)
tm_arb_unloading = st.number_input("TM ARB UNLOADING", value=35.5)
ash_content = st.number_input("Ash Content ARB UNLOADING", value=5.0)
total_sulphur = st.number_input("Total Sulphur ARB UNLOADING", value=0.3)

# **Predict button**
if st.button("Prediksi"):
    input_data = np.array([[gcv_arb_unloading, tm_arb_unloading, ash_content, total_sulphur]])

    # Imputasi dan scaling hanya untuk numerik
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    # Gabungkan kembali dengan Suppliers
    input_final = np.hstack([[supplier_encoded], input_scaled[0]])

    # Prediksi
    prediction = best_model.predict([input_final])
    st.success(f"Prediksi GCV (ARB) LAB: {prediction[0]:.2f}")
