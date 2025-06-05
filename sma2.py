import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go

# --- Fungsi untuk memuat data, model dan label encoder ---
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
st.title("Dashboard Kinerja Karyawan")

# --- Membuat Tab ---
tab1, tab2, tab3, tab4 = st.tabs(["Visualisasi", "Analisis Interaktif", "Prediksi Score", "Prediksi Berdasar Waktu"])

# --- Tab 1: Visualisasi ---
with tab1:
    st.header("Visualisasi Performa Karyawan")
    
    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Karyawan", len(df), delta_color="inverse")
    col2.metric("Rata-rata Skor", round(df['Performance_Score'].mean(), 2), delta_color="inverse")
    col3.metric("Performa Maksimum", df['Performance_Score'].max(), delta_color="inverse")
    
    st.divider()

    # --- Visualisasi Distribusi Performance Score ---
    st.subheader("Distribusi Performance Score")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Jumlah Karyawan per Skor Performa")
        fig1, ax1 = plt.subplots(figsize=(10,10))
        sns.countplot(data=df, x="Performance_Score", ax=ax1, palette="viridis")
        ax1.set_xlabel("Skor Performa")
        ax1.set_ylabel("Jumlah Karyawan")
        st.pyplot(fig1)
        
    with col_b:
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&display=swap');
            .custom-font {
                font-family: 'Roboto', sans-serif;
            }
        </style>
        <h3 class="custom-font" style='font-size: 28px; color: #28476;'>Proporsi Kategori Performa</h3>
        """, unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4,4))
        df['Performance_Score'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=sns.color_palette("Blues", 5), startangle=90)
        ax2.set_ylabel("")
        st.pyplot(fig2)

    st.divider()

    # --- Tren Performa Karyawan ---

    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    df['Hire_Month'] = df['Hire_Date'].dt.to_period("M").dt.to_timestamp()

    col_f, col_g = st.columns(2)
    
    with col_f:
        st.subheader("Tren Skor Performa berdasarkan Waktu")
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

    with col_g:
        st.subheader("Analisis Fitur Numerik terhadap Skor")
        col_e = st.selectbox("Pilih Fitur Numerik:", ['Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled', 'Overtime_Hours', 'Sick_Days', 'Training_Hours', 'Employee_Satisfaction_Score'])
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x="Performance_Score", y=col_e, ax=ax3, palette="magma")
        ax3.set_title(f"Rata-rata {col_e} per Skor Performa")
        st.pyplot(fig3)

    st.divider()

# --- Tab 2: Analisis Interaktif ---
with tab2:
    st.header("Analisis Berdasarkan Filter")
    
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
        fig_a, ax_a = plt.subplots()
        sns.boxplot(data=filtered_df, x="Performance_Score", y="Monthly_Salary", ax=ax_a, palette="Set2")
        ax_a.set_title("Distribusi Gaji per Skor Performa")
        st.pyplot(fig_a)

    with col4:
        fig_b, ax_b = plt.subplots()
        sns.boxplot(data=filtered_df, x="Performance_Score", y="Employee_Satisfaction_Score", ax=ax_b, palette="Pastel1")
        ax_b.set_title("Distribusi Kepuasan per Skor Performa")
        st.pyplot(fig_b)

# --- Tab 3: Prediksi Interaktif ---
with tab3:
    st.header("Prediksi Performance Score Berdasarkan Input")
    
    # --- User Input for Prediction ---
    department = st.selectbox("Department", label_encoders['Department'].classes_)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    age = st.slider("Age", 22, 60, 40)
    job_title = st.selectbox("Job Title", label_encoders['Job_Title'].classes_)
    # hire_date = st.date_input("Hire Date", datetime(2020, 1, 1))
    years_at_company = st.slider("Years at Company", 0, 10, 4)
    education = st.selectbox("Education Level", label_encoders['Education_Level'].classes_)
    monthly_salary = st.number_input("Monthly Salary", 3850.0, 9000.0, 6400.0)
    work_hours = st.slider("Work Hours Per Week", 30, 60, 45)
    projects = st.slider("Projects Handled", 0, 49, 24)
    overtime = st.slider("Overtime Hours", 0, 29, 15)
    sick_days = st.slider("Sick Days", 0, 14, 7)
    remote_freq = st.slider("Remote Work Frequency", 0, 100, 50)
    team_size = st.slider("Team Size", 1, 19, 10)
    training_hours = st.slider("Training Hours", 0, 99, 49)
    promotions = st.slider("Promotions", 0, 2, 1)
    satisfaction = st.slider("Employee Satisfaction Score", 1.0, 5.0, 3.0)
    resigned = st.selectbox("Resigned?", ['False', 'True'])
    
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
    if st.button("Prediksi Performance Score"):
        pred = model.predict(input_df)[0]
        st.success(f"Prediksi Performance Score: **{pred}**")


# --- Tab 4: Prediksi Time Series ---
with tab4:
    st.subheader("Prediksi SMA Rata-rata Skor Performa")

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
