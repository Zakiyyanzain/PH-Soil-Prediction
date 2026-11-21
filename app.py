import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import xgboost as xgb
import lightgbm as lgb

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Sistem Cerdas Prediksi pH Tanah", layout="wide", page_icon="🌱")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    .rec-box {padding: 15px; border-radius: 10px; border-left: 5px solid;}
    .acid {background-color: #ffebee; border-color: #ef5350;}
    .neutral {background-color: #e8f5e9; border-color: #66bb6a;}
    .alkaline {background-color: #e3f2fd; border-color: #42a5f5;}
</style>
""", unsafe_allow_html=True)

st.title("🌱 Smart Soil: Prediksi & Rekomendasi pH Tanah")
st.markdown("Ubah data nutrisi tanah menjadi wawasan yang dapat ditindaklanjuti untuk pertanian.")

# --- Session State ---
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = None
if 'imputer' not in st.session_state:
    st.session_state['imputer'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'model_rmse' not in st.session_state: # Baru: Simpan nilai error model
    st.session_state['model_rmse'] = 0.0

# --- Sidebar ---
st.sidebar.header("⚙️ Panel Kontrol")
uploaded_file = st.sidebar.file_uploader("1. Upload Data Training (Excel)", type=["xlsx"])

@st.cache_data
def load_data(file):
    return pd.read_excel(file, sheet_name='SOIL DATA').dropna(axis=1, how='all')

# --- TAB LAYOUT ---
tab1, tab2 = st.tabs(["📊 Training & Evaluasi Model", "🚜 Prediksi Batch & Manual"])

# =============================================================================
# TAB 1: TRAINING MODEL
# =============================================================================
with tab1:
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            
            if 'pH' in df.columns:
                X = df.drop(columns=['pH', 'ID'], errors='ignore')
                y = df['pH']
                
                st.session_state['feature_names'] = X.columns.tolist()

                X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                imputer = SimpleImputer(strategy='mean')
                scaler = StandardScaler()

                X_train_imputed = imputer.fit_transform(X_train_raw)
                X_test_imputed = imputer.transform(X_test_raw)
                
                X_train = scaler.fit_transform(X_train_imputed)
                X_test = scaler.transform(X_test_imputed)
                
                st.session_state['scaler'] = scaler 
                st.session_state['imputer'] = imputer

                st.success(f"✅ Data Termuat: {len(X_train)} data latih.")
            else:
                st.error("Kolom 'pH' tidak ditemukan.")
                st.stop()
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        st.info("Silakan upload file di sidebar untuk memulai.")

    model_option = st.selectbox(
        "Pilih Algoritma",
        [
            "KNN Regression", "Decision Tree", "Random Forest", "Support Vector Regression (SVR)",
            "Gradient Boosting", "Neural Network (MLP)", "XGBoost Regression", 
            "LightGBM Regression", "Polynomial Regression", "Bayesian Ridge"
        ]
    )
    
    btn_train = st.button("Latih Model", type="primary")

    if btn_train and uploaded_file:
        with st.spinner(f"Sedang melatih {model_option}..."):
            
            # --- LOGIKA MODEL (SAMA SEPERTI SEBELUMNYA) ---
            if model_option == "KNN Regression":
                best_score = -np.inf
                best_model = None
                for k in range(1, 13):
                    knn = KNeighborsRegressor(n_neighbors=k)
                    knn.fit(X_train, y_train)
                    score = knn.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_model = knn
                final_model = best_model
            
            elif model_option == "Decision Tree":
                dt = DecisionTreeRegressor(random_state=42)
                grid = GridSearchCV(dt, {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5]}, cv=5, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_

            elif model_option == "Random Forest":
                rf = RandomForestRegressor(random_state=42)
                grid = GridSearchCV(rf, {'n_estimators': [50, 100]}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_
            
            elif model_option == "Support Vector Regression (SVR)":
                svr = SVR()
                grid = GridSearchCV(svr, {'C': [1, 10], 'kernel': ['rbf']}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_

            elif model_option == "Gradient Boosting":
                gb = GradientBoostingRegressor(random_state=42)
                grid = GridSearchCV(gb, {'n_estimators': [100], 'learning_rate': [0.1]}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_

            elif model_option == "Neural Network (MLP)":
                mlp = MLPRegressor(random_state=42, max_iter=500)
                grid = GridSearchCV(mlp, {'hidden_layer_sizes': [(50,), (100,)]}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_
            
            elif model_option == "XGBoost Regression":
                xg = xgb.XGBRegressor(random_state=42)
                grid = GridSearchCV(xg, {'n_estimators': [100], 'learning_rate': [0.1]}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_
            
            elif model_option == "LightGBM Regression":
                lg = lgb.LGBMRegressor(random_state=42, verbose=-1)
                grid = GridSearchCV(lg, {'n_estimators': [100]}, cv=3, scoring='r2')
                grid.fit(X_train, y_train)
                final_model = grid.best_estimator_
            
            elif model_option == "Bayesian Ridge":
                br = BayesianRidge()
                br.fit(X_train, y_train)
                final_model = br
            
            elif model_option == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2, include_bias=False)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                final_model = Pipeline([('poly', poly), ('linear', model)])
            
            # --- EVALUASI & SIMPAN ERROR ---
            st.session_state['trained_model'] = final_model
            
            y_pred = final_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # [REVISI 1]: Simpan RMSE sebagai nilai toleransi error
            rmse = np.sqrt(mse)
            st.session_state['model_rmse'] = rmse

            c1, c2, c3 = st.columns(3)
            c1.metric("Akurasi (R²)", f"{r2:.4f}")
            c2.metric("Mean Squared Error", f"{mse:.4f}")
            c3.metric("Toleransi Error (RMSE)", f"± {rmse:.4f}", delta="Estimasi Simpangan", delta_color="off")

            fig, ax = plt.subplots(figsize=(8,3))
            sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
            
            st.success(f"Model {model_option} berhasil disimpan!")

# =============================================================================
# TAB 2: FITUR PREDIKSI (DENGAN TOLERANSI)
# =============================================================================
with tab2:
    st.header("🚜 Prediksi pH Tanah")
    
    if st.session_state['trained_model'] is None:
        st.warning("⚠️ Harap latih model terlebih dahulu di Tab 'Training & Evaluasi Model'.")
    else:
        input_method = st.radio("Pilih Metode Input Data:", 
                                ["📂 Upload File (Batch)", "📝 Input Manual (Satu Data)"], 
                                horizontal=True)
        st.markdown("---")

        # --- OPTION A: UPLOAD BATCH ---
        if input_method == "📂 Upload File (Batch)":
            st.subheader("Batch Prediction")
            st.info("Upload file Excel/CSV yang berisi kolom fitur.")
            
            pred_file = st.file_uploader("Upload Data Baru", type=["xlsx", "csv"], key="batch_upload")
            
            if pred_file:
                try:
                    if pred_file.name.endswith('.csv'):
                        df_new = pd.read_csv(pred_file)
                    else:
                        df_new = pd.read_excel(pred_file)
                    
                    jumlah_data = len(df_new)
                    st.write(f"📂 **Status File:** Terdeteksi **{jumlah_data} baris** data.")
                    
                    with st.expander("Lihat Preview Data Upload"):
                        st.dataframe(df_new.head())
                    
                    required_features = st.session_state['feature_names']
                    missing_cols = [col for col in required_features if col not in df_new.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Kolom berikut hilang: {missing_cols}")
                    else:
                        if st.button("Jalankan Prediksi Batch"):
                            X_batch = df_new[required_features]
                            
                            X_batch_imputed = st.session_state['imputer'].transform(X_batch)
                            X_batch_scaled = st.session_state['scaler'].transform(X_batch_imputed)
                            
                            predictions = st.session_state['trained_model'].predict(X_batch_scaled)
                            
                            df_result = df_new.copy()
                            df_result['Predicted_pH'] = predictions
                            
                            st.success("✅ Prediksi Selesai!")
                            st.dataframe(df_result)
                            
                            # --- [REVISI 2]: TAMPILAN TOLERANSI ---
                            st.write("📊 Ringkasan Prediksi:")
                            
                            # Ambil Toleransi dari RMSE Training
                            toleransi = st.session_state['model_rmse']
                            rata_rata_prediksi = predictions.mean()
                            
                            col_s1, col_s2 = st.columns(2)
                            
                            # Jika hanya 1 data, Rata-rata = Nilai itu sendiri
                            label_rata = "Nilai Prediksi pH" if len(predictions) == 1 else "Rata-rata Prediksi pH"
                            
                            col_s1.metric(label_rata, f"{rata_rata_prediksi:.2f}")
                            col_s2.metric("Toleransi Error (Estimasi)", f"± {toleransi:.2f}", 
                                          help="Berdasarkan performa model saat training (RMSE). Nilai asli kemungkinan berada dalam rentang ini.")
                            
                            # Download
                            csv = df_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="💾 Download Hasil (CSV)",
                                data=csv,
                                file_name="hasil_prediksi_ph.csv",
                                mime="text/csv",
                            )

                except Exception as e:
                    st.error(f"Terjadi kesalahan pembacaan file: {e}")

        # --- OPTION B: INPUT MANUAL (DENGAN TOLERANSI) ---
        else:
            st.subheader("Input Manual (Single Prediction)")
            
            if st.session_state['feature_names']:
                input_data = {}
                cols = st.columns(3)
                
                for i, col_name in enumerate(st.session_state['feature_names']):
                    with cols[i % 3]:
                        input_data[col_name] = st.number_input(f"Nilai {col_name}", value=0.0, step=0.1)
                
                if st.button("🔍 Analisis Sampel Tanah Ini"):
                    input_df = pd.DataFrame([input_data])
                    input_imputed = st.session_state['imputer'].transform(input_df)
                    input_scaled = st.session_state['scaler'].transform(input_imputed)
                    
                    prediction = st.session_state['trained_model'].predict(input_scaled)[0]
                    toleransi = st.session_state['model_rmse']
                    
                    st.markdown("---")
                    # Tampilkan Prediksi + Toleransi
                    col_res, col_tol, col_rec = st.columns([1, 1, 2])
                    
                    with col_res:
                        st.metric("Prediksi pH", f"{prediction:.2f}")
                    with col_tol:
                        st.metric("Toleransi (±)", f"{toleransi:.2f}", delta="Margin Error")

                    with col_rec:
                        if prediction < 5.5:
                            status = "ASAM"
                            color_class = "acid"
                            msg = "Tanah Masam. Disarankan pengapuran (Dolomit)."
                        elif 5.5 <= prediction <= 7.5:
                            status = "NETRAL"
                            color_class = "neutral"
                            msg = "Kondisi Optimal. Pertahankan bahan organik."
                        else:
                            status = "BASA"
                            color_class = "alkaline"
                            msg = "Tanah Basa. Berikan pupuk Sulfur."
                        
                        st.markdown(f"""
                        <div class="rec-box {color_class}">
                            <h4>Status: {status}</h4>
                            <p>{msg}</p>
                        </div>
                        """, unsafe_allow_html=True)