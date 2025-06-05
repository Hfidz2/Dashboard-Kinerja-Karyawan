import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go

# --- Fungsi untuk memuat data, model dan label encoder ---

st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Tombol untuk toggle mode
if st.button("🌙" if st.session_state.dark_mode else "☀️"):
    st.session_state.dark_mode = not st.session_state.dark_mode
# --- Mode Tema ---
dark_mode = st.session_state.dark_mode
# dark_mode = st.toggle("Aktifkan Dark Mode", value=True)
# --- Warna berdasarkan tema ---
cmap = "viridis" if not dark_mode else "coolwarm_r"
font_color = "white" if dark_mode else "black"
background_color = "#000000" if dark_mode else "white"
background_color_anakan = "#161b22" if dark_mode else "#FFFFFF"
tab_background = "#0e1117" if dark_mode else "#f0f2f6"
tab_selected = "#1f77b4" if dark_mode else "#d0e0f0"

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {background_color};
            color: {font_color};
        }}
        div[data-testid="stButton"] > button {{
            background-color: {"#0a5285" if dark_mode else '#dddddd'};
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.1em 0.3em;  
            font-size: 0.5rem;     
            #min-width: px;      
            transition: background-color 0.3s ease;
        }}        div[data-testid="stButton"] > button:hover {{
            background-color: {'#155fa0' if dark_mode else '#bbbbbb'};
        }}
        .metric-box {{
            background-color: {background_color_anakan};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: {font_color};
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}
        .metric-title {{
            font-size: 16px;
            font-weight: bold;
            color: {font_color};
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: {font_color};
        }}
        /* --- Tabs Styling --- */
        div[data-baseweb="tab-list"] button {{
            background-color: {tab_background};
            color: {font_color};
            border-radius: 5px;
            padding: 10px;
            margin-right: 5px;
        }}
        div[data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {tab_selected};
            font-weight: bold;
            color: white;
        }}
        label {{
            color: {font_color} !important;
            font-weight: bold;
        }}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_label_encoders():
    return joblib.load("label_encoders.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Extended_Employee_Performance_and_Productivity_Data (1).csv")

# --- Memuat data, model, dan label encoder ---
model = load_model()
label_encoders = load_label_encoders()
df = load_data()

# --- Title of the Dashboard ---
st.markdown("<h1 style='text-align: center; color: font_color;'>Dashboard Kinerja Karyawan</h1><br><br>", unsafe_allow_html=True)
# --- Membuat Tab ---
def tab_header(title, subtitle):
    st.markdown(f"""
    <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
        margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h3 style='margin: 0;'>{title}</h3>
        <p style='margin: 5px 0 0 0;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualisasi", "Analisis Interaktif", "Prediksi Score", "Prediksi Berdasar Waktu", "Korelasi Fitur"])

# --- Tab 1: Visualisasi ---
with tab1:
    tab_header("Visualisasi Performa Karyawan", "Jelajahi data, dapatkan insight, dan prediksi performa karyawan secara interaktif.")

    # --- Metrics  ---
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Jumlah Karyawan</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_score = round(df['Performance_Score'].mean(), 2)
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Rata-rata Skor</div>
                    <div class="metric-value">{avg_score}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            max_score = df['Performance_Score'].max()
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Performa Maksimum</div>
                    <div class="metric-value">{max_score}</div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()


    # --- Visualisasi Distribusi Performance Score ---
    tab_header("Distribusi Performance Score", "Visualisasi distribusi skor performa karyawan.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Jumlah Karyawan per Skor Performa</h4>
        """, unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(10,10))
        sns.countplot(data=df, x="Performance_Score", ax=ax1, palette="viridis")
        ax1.set_xlabel("Skor Performa")
        ax1.set_ylabel("Jumlah Karyawan")
        st.pyplot(fig1)

        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Jumlah Karyawan per Skor Performa</h4>
        """, unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(4,4))
        df['Performance_Score'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=sns.color_palette("Blues", 5), startangle=90)
        ax2.set_ylabel("")
        st.pyplot(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Tren Performa Karyawan ---

    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    df['Hire_Month'] = df['Hire_Date'].dt.to_period("M").dt.to_timestamp()

    col_f, col_g = st.columns(2)
    
    with col_f:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Tren Skor Performa berdasarkan Waktu</h4>
        """, unsafe_allow_html=True)

        mode = st.selectbox("Pilih Mode Tampilan", ["Rata-rata Bulanan", "Per Divisi (Top 3)"], index=0)

        if mode == "Rata-rata Bulanan":
            line_data = df.groupby('Hire_Month')['Performance_Score'].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=line_data, x='Hire_Month', y='Performance_Score', ax=ax4, color='#0E4D92', linewidth=2)
            ax4.set_title("Rata-rata Skor Per Bulan")
            ax4.set_xlabel("Bulan")
            ax4.set_ylabel("Skor")
            st.pyplot(fig4)
        else:
            top_dept = df['Department'].value_counts().head(3).index
            dept_means = {dept: df[df['Department'] == dept].groupby('Hire_Month')['Performance_Score'].mean() for dept in top_dept}
            fig5, ax5 = plt.subplots(figsize=(6, 6))
            # for dept in top_dept:
            #     sns.lineplot(data=dept_means[dept].reset_index(), x='Hire_Month', y='Performance_Score', label=dept, ax=ax5, linewidth=2)

            # Hitung rata-rata akhir setiap garis (nilai terakhir untuk masing-masing divisi)
            last_scores = {dept: data.iloc[-1] for dept, data in dept_means.items()}
            highest_dept = max(last_scores, key=lambda d: last_scores[d])

            for dept in top_dept:
                line_data = dept_means[dept].reset_index()
                if dept == highest_dept:
                    sns.lineplot(data=line_data, x='Hire_Month', y='Performance_Score', label=dept, ax=ax5, linewidth=3, alpha=1.0)
                else:
                    sns.lineplot(data=line_data, x='Hire_Month', y='Performance_Score', label=dept, ax=ax5, linewidth=1.5, alpha=0.4)

            ax5.set_title("Top 3 Divisi")
            ax5.set_xlabel("Bulan")
            ax5.set_ylabel("Skor")
            ax5.legend(title="Divisi")
            st.pyplot(fig5)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_g:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Analisis Fitur Numerik terhadap Skor</h4>
        """, unsafe_allow_html=True)

        col_e = st.selectbox("Pilih Fitur Numerik:", ['Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days', 'Training_Hours', 'Employee_Satisfaction_Score'])
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Performance_Score", y=col_e, ax=ax3, palette="magma")
        ax3.set_title(f"Rata-rata {col_e} per Skor Performa")
        st.pyplot(fig3)

        st.markdown("</div>", unsafe_allow_html=True)
    st.divider()

# --- Tab 2: Analisis Interaktif ---
with tab2:
    tab_header("Analisis Interaktif", "Analisis performa karyawan berdasarkan berbagai filter")
    
    # --- Filters ---
    age_range = st.slider("Usia", int(df['Age'].min()), int(df['Age'].max()), (25, 40))
    salary_range = st.slider("Rentang Gaji Bulanan", float(df['Monthly_Salary'].min()), float(df['Monthly_Salary'].max()), (3850.0, 9000.0))
    satisfaction_range = st.slider("Rentang Skor Kepuasan Karyawan", 1.0, 5.0, (2.0, 4.5))
    education_options = st.multiselect("Pilih Tingkat Pendidikan", options=df['Education_Level'].unique(), default=list(df['Education_Level'].unique()))
    jobtitle_options = st.multiselect("Pilih Jabatan", options=df['Job_Title'].unique(), default=list(df['Job_Title'].unique()))
    department_options = st.multiselect("Pilih Departemen", options=df['Department'].unique(), default=list(df['Department'].unique()))
    
    # --- Filter Applied ---
    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
                     (df['Monthly_Salary'] >= salary_range[0]) & (df['Monthly_Salary'] <= salary_range[1]) &
                     (df['Employee_Satisfaction_Score'] >= satisfaction_range[0]) & (df['Employee_Satisfaction_Score'] <= satisfaction_range[1]) &
                     (df['Education_Level'].isin(education_options)) &
                     (df['Job_Title'].isin(jobtitle_options)) &
                     (df['Department'].isin(department_options))]

    st.dataframe(filtered_df, use_container_width=True)
    
    # --- Boxplots for Performance Analysis ---
    st.subheader("Distribusi Fitur Berdasarkan Performance Score")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Analisis Fitur Numerik terhadap Skor</h4>
        """, unsafe_allow_html=True)

        fig_a, ax_a = plt.subplots()
        sns.boxplot(data=filtered_df, x="Performance_Score", y="Monthly_Salary", ax=ax_a, palette="Set2")
        ax_a.set_title("Distribusi Gaji per Skor Performa")
        st.pyplot(fig_a)

        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
                margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h4 style='margin-top: 0;'>Analisis Fitur Numerik terhadap Skor</h4>
        """, unsafe_allow_html=True)

        fig_b, ax_b = plt.subplots()
        sns.boxplot(data=filtered_df, x="Performance_Score", y="Employee_Satisfaction_Score", ax=ax_b, palette="Pastel1")
        ax_b.set_title("Distribusi Kepuasan per Skor Performa")
        st.pyplot(fig_b)

        st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 3: Prediksi Interaktif ---
with tab3:
    tab_header("Prediksi Performance Score", "Masukkan data karyawan untuk memprediksi skor performa mereka.")

    # === Informasi Dasar ===
    with st.container():
        st.markdown("### 🧾 Informasi Dasar")
        col1, col2, col3 = st.columns(3)
        with col1:
            department = st.selectbox("Department", label_encoders['Department'].classes_)
            gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        with col2:
            job_title = st.selectbox("Job Title", label_encoders['Job_Title'].classes_)
            age = st.slider("Age", 22, 60, 40)
        with col3:
            education = st.selectbox("Education Level", label_encoders['Education_Level'].classes_)
            monthly_salary = st.number_input("Monthly Salary", 3850.0, 9000.0, 6400.0)

    # === Status & Lama Bekerja ===
    with st.container():
        st.markdown("### 🏢 Status & Lama Bekerja")
        col1, col2, col3 = st.columns(3)
        with col1:
            years_at_company = st.slider("Years at Company", 0, 10, 4)
            promotions = st.slider("Promotions", 0, 2, 1)
        with col2:
            team_size = st.slider("Team Size", 1, 19, 10)
            sick_days = st.slider("Sick Days", 0, 14, 7)
        with col3:
            training_hours = st.slider("Training Hours", 0, 99, 49)
            resigned = st.selectbox("Resigned?", ['False', 'True'])

    # === Aktivitas & Output ===
    with st.container():
        st.markdown("### 🕒 Aktivitas & Output")
        col1, col2, col3 = st.columns(3)
        with col1:
            work_hours = st.slider("Work Hours Per Week", 30, 60, 45)
        with col2:
            overtime = st.slider("Overtime Hours", 0, 29, 15)
        with col3:
            projects = st.slider("Projects Handled", 0, 49, 24)

        col4, col5, _ = st.columns(3)
        with col4:
            remote_freq = st.slider("Remote Work Frequency (%)", 0, 100, 50)
        with col5:
            satisfaction = st.slider("Employee Satisfaction Score", 1.0, 5.0, 3.0)

    
    # --- Prepare input for prediction ---
    input_dict = {
        'Department': label_encoders['Department'].transform([department])[0],
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Age': age,
        'Job_Title': label_encoders['Job_Title'].transform([job_title])[0],
        'Years_At_Company': years_at_company,
        'Education_Level': label_encoders['Education_Level'].transform([education])[0],
        'Monthly_Salary': monthly_salary,
        'Work_Hours_Per_Week': work_hours,
        'Projects_Handled': projects,
        'Overtime_Hours': overtime,
        'Sick_Days': sick_days,
        'Remote_Work_Frequency': remote_freq,
        'Team_Size': team_size,
        'Training_Hours': training_hours,
        'Promotions': promotions,
        'Employee_Satisfaction_Score': satisfaction,
        'Resigned': label_encoders['Resigned'].transform([resigned])[0]
    }
    input_df = pd.DataFrame([input_dict])

    # --- Prediction ---
    with st.container():
        st.markdown("<h4 style='color:{background_color_anakan};'>Hasil Prediksi</h4>", unsafe_allow_html=True)
    
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #00ff9f;
            color: black;
            font-weight: bold;

            transition: background-color 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #00cc7a;
            color: white;
        }
        div.stAlert-success, div.stAlert-success > div {
            color: black !important;
            background-color: #00ff9f !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.button("Prediksi Performance Score"):
            pred = model.predict(input_df)[0]
            st.markdown(f"""
                <div style="
                    background-color: #00ff9f;
                    padding: 10px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 18px;
                    color: black;
                    display: inline-block;
                    max-width: 300px;
                    text-align: center;
                ">
                    🎯 Prediksi Performance Score: <strong>{pred}</strong>
                </div>
            """, unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 4: Prediksi Time Series ---
with tab4:
    tab_header("Prediksi SMA Rata-rata Skor Performa", "Gunakan metode SMA untuk memprediksi skor performa karyawan ke depan.")

    forecast_months = st.selectbox("Pilih periode prediksi:", [6, 12, 24], index=1)

    # --- Persiapan data ---
    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    df['Month'] = df['Hire_Date'].dt.to_period('M')
    monthly_avg = df.groupby('Month')['Performance_Score'].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].dt.to_timestamp()

    # Hitung SMA historis
    window_size = 4
    exclude_month = pd.Timestamp('2024-09-01')
    monthly_avg = monthly_avg[monthly_avg['Month'] != exclude_month]
    monthly_avg['SMA'] = monthly_avg['Performance_Score'].rolling(window=window_size).mean()

    # --- Prediksi ke depan berbasis SMA bergerak ---
    sma_source = monthly_avg['Performance_Score'].tolist()
    predicted_scores = []

    np.random.seed(42)

    for _ in range(forecast_months):
        if len(sma_source) < window_size:
            sma = np.mean(sma_source)
        else:
            sma = np.mean(sma_source[-window_size:])

        sma += np.random.normal(0, 0.015)
        predicted_scores.append(sma)
        sma_source.append(sma)  # tambahkan ke list untuk hitung prediksi berikutnya

    # --- Siapkan tanggal untuk prediksi ---
    future_dates = pd.date_range(
        start=monthly_avg['Month'].iloc[-1] + pd.offsets.MonthBegin(),
        periods=forecast_months,
        freq='MS'
    )

    df_prediction = pd.DataFrame({
        'Month': future_dates,
        'Predicted_Score': predicted_scores
    })

    # --- Visualisasi ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_avg['Month'], y=monthly_avg['SMA'],
                             mode='lines+markers', name='SMA Historis',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_prediction['Month'], y=df_prediction['Predicted_Score'],
                             mode='lines+markers', name='Prediksi',
                             line=dict(color='orange')))

    fig.update_layout(title='Prediksi SMA Rata-rata Skor Performa',
                      xaxis_title='Month',
                      yaxis_title='Predicted_Score',
                      legend_title='Label',
                      width=900)

    st.plotly_chart(fig)

    # --- Tambahan Tab 5: Dark/Light Mode dan Heatmap Korelasi ---

with tab5:
    tab_header("Korelasi Fitur", "lihat korelasi antar fitur.")

    # --- Heatmap Korelasi ---
    numeric_df = df.select_dtypes(include=[np.number])  # hanya ambil kolom numerik
    corr = numeric_df.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap=cmap, fmt=".2f", ax=ax_corr, linewidths=0.5)
    ax_corr.set_title("Matriks Korelasi", fontsize=16, color=font_color)
    st.pyplot(fig_corr)

