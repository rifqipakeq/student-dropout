
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(102,126,234,0.4);
    }
    .main-header h1 { font-size: 2.2rem; margin-bottom: 0.5rem; }
    .main-header p  { font-size: 1rem; opacity: 0.9; margin: 0; }

    /* Result cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
    }
    .result-dropout {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .result-graduate {
        background: linear-gradient(135deg, #1dd1a1, #10ac84);
        color: white;
    }

    /* Section divider */
    .section-title {
        color: #5f27cd;
        font-size: 1rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-left: 4px solid #5f27cd;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* Metric badge */
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }

    /* Info box */
    .info-box {
        background: #f8f9ff;
        border: 1px solid #e0e7ff;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    """Load model dan preprocessing artifacts (cached)."""
    try:
        model    = joblib.load('model.pkl')
        scaler   = joblib.load('scaler.pkl')
        le       = joblib.load('label_encoder.pkl')
        features = joblib.load('feature_names.pkl')
        return model, scaler, le, features
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}\n")
        st.stop()

model, scaler, le, feature_names = load_artifacts()

# Header
st.markdown("""
<div class="main-header">
    <h1>Student Dropout Predictor</h1>
    <p>Sistem prediksi risiko dropout mahasiswa berbasis Machine Learning<br>
</div>
""", unsafe_allow_html=True)

# Sidebar — Model Info
with st.sidebar:
    st.markdown("### Panduan Penggunaan")
    st.markdown("""
    1. Isi form data mahasiswa
    2. Klik **Prediksi**
    3. Lihat hasil & probabilitas
    4. Ambil tindakan berdasarkan risiko
    """)

    st.markdown("---")
    st.markdown("### Interpretasi Hasil")
    st.markdown("""
    - **Risiko Rendah** (<30%): Monitor rutin
    - **Risiko Sedang** (30-60%): Bimbingan akademik
    - **Risiko Tinggi** (>60%): Intervensi segera
    """)

# Input Form
tab1, tab2 = st.tabs(["Input Data Mahasiswa", "Dashboard Analitik"])

with tab1:
    with st.form("prediction_form"):
        # Informasi Demografis 
        st.markdown('<p class="section-title"> Informasi Demografis</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            age_at_enrollment = st.slider("Usia saat mendaftar", 17, 70, 21)
            gender = st.selectbox("Jenis Kelamin", [("Perempuan", 0), ("Laki-laki", 1)],
                                  format_func=lambda x: x[0])

        with col2:
            marital_status = st.selectbox("Status Pernikahan",
                [(1,"Lajang"),(2,"Menikah"),(3,"Duda/Janda"),(4,"Bercerai"),(5,"Faktual"),(6,"Terpisah")],
                format_func=lambda x: x[1])
            displaced = st.selectbox("Displaced (pindah domisili)?",
                [(0,"Tidak"),(1,"Ya")], format_func=lambda x: x[1])

        with col3:
            international = st.selectbox("Mahasiswa Internasional?",
                [(0,"Tidak"),(1,"Ya")], format_func=lambda x: x[1])
            special_needs = st.selectbox("Kebutuhan Khusus?",
                [(0,"Tidak"),(1,"Ya")], format_func=lambda x: x[1])

        # Akademik 
        st.markdown('<p class="section-title"> Latar Belakang Akademik</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            prev_qual = st.selectbox("Kualifikasi Sebelumnya",
                list(range(1, 18)), format_func=lambda x: f"Kode {x}")
            prev_qual_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 130.0, step=0.5)

        with col2:
            admission_grade = st.slider("Nilai Penerimaan", 0.0, 200.0, 130.0, step=0.5)
            application_mode = st.selectbox("Mode Pendaftaran",
                [1,2,5,7,10,15,16,17,18,26,27,39,42,43,44,51,53,57],
                format_func=lambda x: f"Mode {x}")

        with col3:
            application_order = st.slider("Urutan Pilihan Pendaftaran", 1, 9, 1)
            course = st.number_input("Kode Program Studi", min_value=1, value=9254)

        # Keuangan 
        st.markdown('<p class="section-title">Status Keuangan</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            tuition_up_to_date = st.selectbox("Biaya Kuliah Terkini?",
                [(0,"Belum Lunas"),(1,"Lunas")], format_func=lambda x: x[1])
        with col2:
            debtor = st.selectbox("Status Debitur?",
                [(0,"Bukan Debitur"),(1,"Debitur")], format_func=lambda x: x[1])
        with col3:
            scholarship = st.selectbox("Penerima Beasiswa?",
                [(0,"Tidak"),(1,"Ya")], format_func=lambda x: x[1])

        # Performa Akademik Semester 1 
        st.markdown('<p class="section-title"> Performa Semester 1</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cu1_credited   = st.number_input("SKS Dikreditkan", 0, 20, 0, key="c1cr")
        with col2:
            cu1_enrolled   = st.number_input("SKS Terdaftar", 0, 20, 6, key="c1en")
        with col3:
            cu1_evaluations = st.number_input("SKS Dievaluasi", 0, 30, 6, key="c1ev")
        with col4:
            cu1_approved   = st.number_input("SKS Disetujui", 0, 20, 5, key="c1ap")

        col1, col2, col3 = st.columns(3)
        with col1:
            cu1_grade = st.slider("Rata-rata Nilai Sem 1", 0.0, 20.0, 12.0, step=0.1)
        with col2:
            cu1_no_eval = st.number_input("SKS Tanpa Evaluasi Sem 1", 0, 20, 0, key="c1ne")
        with col3:
            pass  # spacer

        # Performa Akademik Semester 2 
        st.markdown('<p class="section-title"> Performa Semester 2</p>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cu2_credited   = st.number_input("SKS Dikreditkan", 0, 20, 0, key="c2cr")
        with col2:
            cu2_enrolled   = st.number_input("SKS Terdaftar", 0, 20, 6, key="c2en")
        with col3:
            cu2_evaluations = st.number_input("SKS Dievaluasi", 0, 30, 6, key="c2ev")
        with col4:
            cu2_approved   = st.number_input("SKS Disetujui", 0, 20, 5, key="c2ap")

        col1, col2, col3 = st.columns(3)
        with col1:
            cu2_grade = st.slider("Rata-rata Nilai Sem 2", 0.0, 20.0, 12.0, step=0.1)
        with col2:
            cu2_no_eval = st.number_input("SKS Tanpa Evaluasi Sem 2", 0, 20, 0, key="c2ne")

        # Latar Belakang Orang Tua & Makroekonomi 
        st.markdown('<p class="section-title">Konteks Sosial & Ekonomi Makro</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            mothers_qual = st.selectbox("Kualifikasi Ibu", list(range(1, 45)),
                                        format_func=lambda x: f"Level {x}")
            fathers_qual = st.selectbox("Kualifikasi Ayah", list(range(1, 45)),
                                        format_func=lambda x: f"Level {x}")

        with col2:
            mothers_occ  = st.selectbox("Pekerjaan Ibu", list(range(0, 200, 10)),
                                        format_func=lambda x: f"Kode {x}")
            fathers_occ  = st.selectbox("Pekerjaan Ayah", list(range(0, 200, 10)),
                                        format_func=lambda x: f"Kode {x}")
            nacionality  = st.number_input("Kode Kewarganegaraan", 1, 200, 1)

        with col3:
            daytime      = st.selectbox("Waktu Kuliah", [(1,"Siang"),(0,"Malam")],
                                        format_func=lambda x: x[1])
            unemployment_rate = st.slider("Tingkat Pengangguran (%)", 0.0, 25.0, 10.8, step=0.1)
            inflation_rate    = st.slider("Tingkat Inflasi (%)", -5.0, 10.0, 1.4, step=0.1)
            gdp               = st.slider("GDP Growth (%)", -5.0, 5.0, 1.74, step=0.01)

        # Submit Button 
        submitted = st.form_submit_button("Prediksi Risiko Dropout", use_container_width=True)

    # Prediksi & Hasil
    if submitted:
        # Feature engineering
        apr1 = cu1_approved / cu1_enrolled if cu1_enrolled > 0 else 0
        apr2 = cu2_approved / cu2_enrolled if cu2_enrolled > 0 else 0
        avg_g = (cu1_grade + cu2_grade) / 2

        input_dict = {
            'Marital_status': marital_status[0],
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': daytime[0],
            'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade,
            'Nacionality': nacionality,
            'Mothers_qualification': mothers_qual,
            'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ,
            'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade,
            'Displaced': displaced[0],
            'Educational_special_needs': special_needs[0],
            'Debtor': debtor[0],
            'Tuition_fees_up_to_date': tuition_up_to_date[0],
            'Gender': gender[1],
            'Scholarship_holder': scholarship[0],
            'Age_at_enrollment': age_at_enrollment,
            'International': international[0],
            'Curricular_units_1st_sem_credited': cu1_credited,
            'Curricular_units_1st_sem_enrolled': cu1_enrolled,
            'Curricular_units_1st_sem_evaluations': cu1_evaluations,
            'Curricular_units_1st_sem_approved': cu1_approved,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
            'Curricular_units_2nd_sem_credited': cu2_credited,
            'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
            'Curricular_units_2nd_sem_evaluations': cu2_evaluations,
            'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade,
            'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp,
            'approval_rate_sem1': apr1,
            'approval_rate_sem2': apr2,
            'avg_grade': avg_g,
        }

        input_df = pd.DataFrame([input_dict])[feature_names]
        input_scaled = scaler.transform(input_df)

        pred_label = le.inverse_transform(model.predict(input_scaled))[0]
        pred_prob  = model.predict_proba(input_scaled)[0]
        dropout_prob  = pred_prob[0] * 100  
        graduate_prob = pred_prob[1] * 100

        st.markdown("---")
        st.markdown("## Hasil Prediksi")

        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            if pred_label == 'Dropout':
                st.markdown(f"""
                <div class="result-card result-dropout">
                    ⚠️ BERISIKO DROPOUT<br>
                    <span style="font-size:2rem; font-weight:800">{dropout_prob:.1f}%</span><br>
                    <small>probabilitas dropout</small>
                </div>
                """, unsafe_allow_html=True)
                risk_level = "🔴 Tinggi" if dropout_prob > 60 else "🟡 Sedang"
                st.warning(f"**Level Risiko:** {risk_level}")
                st.error("**Rekomendasi:** Segera hubungi mahasiswa untuk bimbingan akademik dan evaluasi kendala finansial/personal.")
            else:
                st.markdown(f"""
                <div class="result-card result-graduate">
                    ✅ DIPREDIKSI LULUS<br>
                    <span style="font-size:2rem; font-weight:800">{graduate_prob:.1f}%</span><br>
                    <small>probabilitas lulus</small>
                </div>
                """, unsafe_allow_html=True)
                st.success("**Rekomendasi:** Pertahankan performa. Monitor berkala tetap diperlukan.")

        with col_gauge:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=dropout_prob,
                number={'suffix': "%", 'font': {'size': 36}},
                title={'text': "Probabilitas Dropout", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#e74c3c" if dropout_prob > 50 else "#f39c12" if dropout_prob > 30 else "#2ecc71"},
                    'steps': [
                        {'range': [0, 30],  'color': "#d5f5e3"},
                        {'range': [30, 60], 'color': "#fdebd0"},
                        {'range': [60, 100],'color': "#fadbd8"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Detail probabilitas
        st.markdown("### Detail Probabilitas")
        fig_bar = go.Figure([
            go.Bar(x=['Dropout', 'Graduate'],
                   y=[dropout_prob, graduate_prob],
                   marker_color=['#e74c3c', '#2ecc71'],
                   text=[f'{dropout_prob:.1f}%', f'{graduate_prob:.1f}%'],
                   textposition='auto',
                   width=0.4)
        ])
        fig_bar.update_layout(yaxis_title="Probabilitas (%)", yaxis_range=[0, 110],
                              height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 2: Dashboard Analitik
with tab2:
    st.markdown("### Ringkasan Performa Model")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "91.05%", "")
    col2.metric("ROC-AUC",   "0.9566",  "")
    col3.metric("Precision (Dropout)", "92%",  "")
    col4.metric("Recall (Dropout)",    "84%",  "")

    st.markdown("---")

    # Feature importance
    fi_data = {
        'Curricular_units_2nd_sem_approved':   25.3,
        'Curricular_units_1st_sem_approved':   13.3,
        'Curricular_units_2nd_sem_grade':      12.0,
        'avg_grade':                            8.2,
        'Curricular_units_1st_sem_grade':       7.5,
        'Tuition_fees_up_to_date':              5.9,
        'approval_rate_sem2':                   4.1,
        'Age_at_enrollment':                    3.2,
        'Scholarship_holder':                   2.4,
        'Course':                               2.4,
    }
    df_fi = pd.DataFrame({'Feature': list(fi_data.keys()),
                          'Importance (%)': list(fi_data.values())})
    df_fi = df_fi.sort_values('Importance (%)')

    fig_fi = px.bar(df_fi, x='Importance (%)', y='Feature', orientation='h',
                    color='Importance (%)', color_continuous_scale='Viridis',
                    title='Top 10 Fitur Terpenting')
    fig_fi.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("### 💡 Insight Utama")
    insights = [
        ("📘", "SKS yang disetujui di semester 1 & 2", "Mahasiswa yang tidak lulus banyak SKS sangat berisiko dropout."),
        ("💰", "Status pembayaran biaya kuliah", "Mahasiswa yang menunggak pembayaran memiliki dropout rate 3x lebih tinggi."),
        ("📉", "Nilai rata-rata akademik", "Gap nilai antara Dropout dan Graduate sangat signifikan."),
        ("🎓", "Penerima beasiswa", "Penerima beasiswa memiliki dropout rate lebih rendah ."),
        ("📅", "Usia saat mendaftar", "Mahasiswa yang mendaftar di usia lebih tua cenderung memiliki risiko dropout lebih tinggi."),
    ]
    for icon, title, desc in insights:
        st.markdown(f"**{icon} {title}:** {desc}")
