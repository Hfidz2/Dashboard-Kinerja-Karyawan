import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image

# --- Fungsi untuk memuat data, model dan label encoder ---

st.set_page_config(page_title="Employee Performance Dashboard", layout="wide")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Tombol untuk toggle mode
if st.button("🌙" if st.session_state.dark_mode else "☀️"):
    st.session_state.dark_mode = not st.session_state.dark_mode

dark_mode = st.session_state.dark_mode
cmap = "Viridis" if not dark_mode else "RdBu"
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

model = load_model()
label_encoders = load_label_encoders()
df = load_data()

st.markdown("<h1 style='text-align: center;'>Dashboard Kinerja Karyawan</h1><br><br>", unsafe_allow_html=True)

def tab_header(title, subtitle):
    st.markdown(f"""
    <div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; 
        margin-bottom: 20px; background-color: {background_color_anakan}; color: {font_color};'>
        <h3 style='margin: 0;'>{title}</h3>
        <p style='margin: 5px 0 0 0;'>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Visualisasi", "Analisis Interaktif", "Prediksi Score", "Prediksi Berdasar Waktu", "Fitur Penting"])

# --- Tab 1 ---
with tab1:
    st.subheader("Visualisasi Performa Karyawan")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Karyawan", len(df))
    with col2:
        st.metric("Rata-rata Skor Performa", round(df['Performance_Score'].mean(), 2))
    with col3:
        st.metric("Rata-rata Skor Kepuasan", round(df['Employee_Satisfaction_Score'].mean(),2))
    st.divider()

    st.subheader("Tingkat Kepuasan per Departemen")
    dept_performance = df.groupby("Department")["Employee_Satisfaction_Score"].mean().reset_index()
    dept_performance = dept_performance.sort_values(by="Employee_Satisfaction_Score", ascending=False)
    fig1= px.bar(
        dept_performance,
        x="Employee_Satisfaction_Score",
        y="Department",
        orientation='h',
        title="Rata-rata Skor Kepuasan per Departemen",
        color="Employee_Satisfaction_Score",
        color_continuous_scale="Viridis"
        )
    fig1.update_layout(
        width=400,
        height=350,
        paper_bgcolor=background_color_anakan,
        plot_bgcolor=background_color_anakan,
        font=dict(color=font_color),
        yaxis=dict(autorange="reversed") 
        )
    st.plotly_chart(fig1)
    st.divider()

        
        
    st.subheader("Tingkat Performa per Departemen")
    dept_performance = df.groupby("Department")["Performance_Score"].mean().reset_index()
    dept_performance = dept_performance.sort_values(by="Performance_Score", ascending=False)
    fig2= px.bar(
        dept_performance,
        x="Performance_Score",
        y="Department",
        orientation='h',
        title="Rata-rata Skor Performa per Departemen",
        color="Performance_Score",
        color_continuous_scale="Viridis"
        )
    fig2.update_layout(
        width=400,
        height=350,
        paper_bgcolor=background_color_anakan,
        plot_bgcolor=background_color_anakan,
        font=dict(color=font_color),
        yaxis=dict(autorange="reversed") 
        )
    st.plotly_chart(fig2)
    st.divider()

    st.subheader("Radar Chart: Kompetensi & Produktivitas Karyawan")
    radar_columns = ['Projects_Handled', 'Training_Hours', 'Work_Hours_Per_Week', 'Overtime_Hours', 'Sick_Days']
    radar_data = df.groupby(['Job_Title', 'Department'])[radar_columns].mean().reset_index()
    job_options = radar_data['Job_Title'].unique()
    selected_job = st.selectbox("Pilih Job Title", job_options)
    filtered_data = radar_data[radar_data['Job_Title'] == selected_job]
    import plotly.graph_objects as go
    fig = go.Figure()
    for _, row in filtered_data.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=row[radar_columns].values,
            theta=radar_columns,
            fill='toself',
            name=row['Department']
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f"Radar Chart untuk {selected_job}",
    )
    st.plotly_chart(fig)

# --- Tab 2 ---
with tab2:
    st.subheader("Analisis Interaktif")
    age_range = st.slider("Usia", int(df['Age'].min()), int(df['Age'].max()), (25, 40))
    salary_range = st.slider("Rentang Gaji Bulanan", float(df['Monthly_Salary'].min()), float(df['Monthly_Salary'].max()), (3850.0, 9000.0))
    satisfaction_range = st.slider("Rentang Skor Kepuasan Karyawan", 1.0, 5.0, (2.0, 4.5))
    education_options = st.multiselect("Pilih Tingkat Pendidikan", options=df['Education_Level'].unique(), default=list(df['Education_Level'].unique()))
    jobtitle_options = st.multiselect("Pilih Jabatan", options=df['Job_Title'].unique(), default=list(df['Job_Title'].unique()))
    department_options = st.multiselect("Pilih Departemen", options=df['Department'].unique(), default=list(df['Department'].unique()))

    filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
                     (df['Monthly_Salary'] >= salary_range[0]) & (df['Monthly_Salary'] <= salary_range[1]) &
                     (df['Employee_Satisfaction_Score'] >= satisfaction_range[0]) & (df['Employee_Satisfaction_Score'] <= satisfaction_range[1]) &
                     (df['Education_Level'].isin(education_options)) &
                     (df['Job_Title'].isin(jobtitle_options)) &
                     (df['Department'].isin(department_options))]

    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Distribusi Fitur Berdasarkan Performance Score")
    fig_a = px.box(filtered_df, x="Performance_Score", y="Monthly_Salary", title="Distribusi Gaji per Skor Performa")
    fig_a.update_layout(paper_bgcolor=background_color_anakan, plot_bgcolor=background_color_anakan, font=dict(color=font_color))
    st.plotly_chart(fig_a)

    fig_b = px.box(filtered_df, x="Performance_Score", y="Employee_Satisfaction_Score", title="Distribusi Kepuasan per Skor Performa")
    fig_b.update_layout(paper_bgcolor=background_color_anakan, plot_bgcolor=background_color_anakan, font=dict(color=font_color))
    st.plotly_chart(fig_b)

# --- Tab 3 ---
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
            pred = model.predict(input_df)
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


# --- Tab 4 ---
with tab4:
    st.subheader("Prediksi SMA Rata-rata Skor Performa")
    forecast_months = st.selectbox("Pilih periode prediksi:", [6, 12, 24], index=1)

    df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])
    df['Month'] = df['Hire_Date'].dt.to_period('M')
    monthly_avg = df.groupby('Month')['Performance_Score'].mean().reset_index()
    monthly_avg['Month'] = monthly_avg['Month'].dt.to_timestamp()
    exclude_month = pd.Timestamp('2024-09-01')
    monthly_avg = monthly_avg[monthly_avg['Month'] != exclude_month]

    window_size = 4
    monthly_avg['SMA'] = monthly_avg['Performance_Score'].rolling(window=window_size).mean()
    sma_source = monthly_avg['Performance_Score'].tolist()
    predicted_scores = []
    np.random.seed(42)
    for _ in range(forecast_months):
        sma = np.mean(sma_source[-window_size:]) if len(sma_source) >= window_size else np.mean(sma_source)
        sma += np.random.normal(0, 0.015)
        predicted_scores.append(sma)
        sma_source.append(sma)

    future_dates = pd.date_range(start=monthly_avg['Month'].iloc[-1] + pd.offsets.MonthBegin(), periods=forecast_months, freq='MS')
    df_prediction = pd.DataFrame({'Month': future_dates, 'Predicted_Score': predicted_scores})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_avg['Month'], y=monthly_avg['SMA'], mode='lines+markers', name='SMA Historis', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_prediction['Month'], y=df_prediction['Predicted_Score'], mode='lines+markers', name='Prediksi', line=dict(color='orange')))
    fig.update_layout(title='Prediksi SMA Rata-rata Skor Performa', xaxis_title='Month', yaxis_title='Predicted_Score',
                      legend_title='Label', width=900, plot_bgcolor=background_color_anakan, paper_bgcolor=background_color_anakan,
                      font=dict(color=font_color))
    st.plotly_chart(fig)

# --- Tab 5 ---
with tab5:
    st.session_state.selected_tab = "Fitur Penting"

    with st.container():
        st.subheader("Fitur-Fitur Penting dalam Prediksi")
        importances = model.feature_importances_
        feature_names = input_df.columns.tolist()
        scaled_importances = 10 * (importances - np.min(importances)) / (np.max(importances) - np.min(importances))
        imp_df = pd.DataFrame({"Fitur": feature_names, "Pentingnya": scaled_importances})
        imp_df = imp_df.sort_values("Pentingnya", ascending=True)

        fig_imp = px.bar(imp_df, x="Pentingnya", y="Fitur", orientation="h",
                         title="Pentingnya Setiap Fitur dalam Model (Skala 1-10)",
                         text="Pentingnya",
                         color_discrete_sequence=["#FF6361"])
        fig_imp.update_layout(paper_bgcolor=background_color_anakan,
                              plot_bgcolor=background_color_anakan,
                              font=dict(color=font_color),
                              xaxis=dict(title="Skor Penting (1-10)"))
        fig_imp.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_imp, use_container_width=True)

    with st.container():
        st.subheader("WordCloud Feature Importance")
        word_freq = dict(zip(imp_df["Fitur"], imp_df["Pentingnya"]))
        wc = WordCloud(
            background_color=background_color if not dark_mode else "#161b22",
            colormap="RdBu" if dark_mode else "viridis",
            width=800,
            height=400
        ).generate_from_frequencies(word_freq)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        fig_wc.patch.set_facecolor(background_color_anakan)

        st.pyplot(fig_wc)

    with st.container():
        st.subheader("Donut Chart Feature Importance")
        fig_donut = px.pie(
            imp_df,
            names="Fitur",
            values="Pentingnya",
            hole=0.4,
            title="Distribusi Pentingnya Fitur (Skala 1–10)",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig_donut.update_layout(
            paper_bgcolor=background_color_anakan,
            plot_bgcolor=background_color_anakan,
            font=dict(color=font_color)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with st.container():
        st.subheader("Top 5 Fitur Paling Penting")

        top5_df = imp_df.sort_values("Pentingnya", ascending=False).head(5)

        fig_top5 = px.bar(
            top5_df,
            x="Pentingnya",
            y="Fitur",
            orientation="h",
            title="Top 5 Fitur Penting (Skala 1–10)",
            text="Pentingnya",
            color_discrete_sequence=["#FF6361"]
        )
        fig_top5.update_layout(
            paper_bgcolor=background_color_anakan,
            plot_bgcolor=background_color_anakan,
            font=dict(color=font_color),
            xaxis=dict(title="Skor Penting (1–10)")
        )
        fig_top5.update_traces(texttemplate='%{text:.1f}', textposition='outside')

        st.plotly_chart(fig_top5, use_container_width=True)


