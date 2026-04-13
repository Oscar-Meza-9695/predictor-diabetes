import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="centered",
)

st.markdown("""
<style>
    .result-box-positive {
        background-color: #FEE2E2;
        border-left: 5px solid #DC2626;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .result-box-negative {
        background-color: #DCFCE7;
        border-left: 5px solid #16A34A;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 1rem 0 0.5rem 0;">
    <h1>🩺 Predictor de Diabetes</h1>
    <p style="color: #6B7280; font-size: 0.95rem;">
        Modelo entrenado con FLAML (AutoML) · Dataset Pima Indians Diabetes
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─── Entrenar modelo (se cachea, solo corre una vez) ───
@st.cache_resource
def entrenar_modelo():
    from flaml import AutoML
    from sklearn.model_selection import train_test_split

    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
    df = pd.read_csv(url, names=cols)

    # Preprocesamiento
    cols_fix = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for col in cols_fix:
        df[col] = df[col].replace(0, np.nan)
        m0 = df.loc[df['Outcome']==0, col].median()
        m1 = df.loc[df['Outcome']==1, col].median()
        df.loc[(df['Outcome']==0) & df[col].isna(), col] = m0
        df.loc[(df['Outcome']==1) & df[col].isna(), col] = m1

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    automl = AutoML()
    automl.fit(
        X_train=X_train, y_train=y_train,
        task='classification', metric='roc_auc',
        time_budget=60, seed=42, verbose=0,
        estimator_list=['lgbm','xgboost','rf','extra_tree','lrl1'],
    )
    return automl

with st.spinner('Cargando modelo... (solo la primera vez, ~1 minuto)'):
    modelo = entrenar_modelo()

st.success(f"✅ Modelo listo: **{modelo.best_estimator.upper()}**")

# ─── Formulario ───
st.markdown("### 📋 Datos del paciente")

col1, col2 = st.columns(2)
with col1:
    pregnancies   = st.number_input("Número de embarazos", 0, 20, 2, 1)
    glucose       = st.number_input("Glucosa en plasma (mg/dL)", 0, 300, 120, 1)
    blood_pressure= st.number_input("Presión diastólica (mmHg)", 0, 150, 72, 1)
    skin_thickness= st.number_input("Grosor pliegue cutáneo (mm)", 0, 100, 23, 1)
with col2:
    insulin = st.number_input("Insulina sérica (μU/mL)", 0, 900, 85, 1)
    bmi     = st.number_input("IMC (kg/m²)", 0.0, 70.0, 26.5, 0.1)
    dpf     = st.number_input("Función pedigrí de diabetes", 0.000, 3.000, 0.351, 0.001, format="%.3f")
    age     = st.number_input("Edad (años)", 1, 120, 33, 1)

st.markdown("<br>", unsafe_allow_html=True)
predecir = st.button("🔍 Generar predicción", type="primary", use_container_width=True)

if predecir:
    datos = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                           insulin, bmi, dpf, age]],
                         columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                                  'Insulin','BMI','DiabetesPedigreeFunction','Age'])
    prediccion = modelo.predict(datos)[0]
    probs      = modelo.predict_proba(datos)[0]
    prob_neg, prob_pos = probs[0], probs[1]

    st.divider()
    st.markdown("### 📊 Resultado")

    if prediccion == 1:
        st.markdown(f"""
        <div class="result-box-positive">
            <h3 style="color:#DC2626; margin:0 0 0.5rem 0;">⚠️ Alto riesgo de diabetes detectado</h3>
            <p style="margin:0; color:#7F1D1D;">El modelo indica que el paciente <strong>podría tener diabetes</strong>. Se recomienda consultar con un médico.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-negative">
            <h3 style="color:#16A34A; margin:0 0 0.5rem 0;">✅ Bajo riesgo de diabetes</h3>
            <p style="margin:0; color:#14532D;">El modelo indica que el paciente <strong>probablemente no tiene diabetes</strong>. Mantén hábitos saludables.</p>
        </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Sin diabetes (0)", f"{prob_neg*100:.1f}%")
    c2.metric("Con diabetes (1)", f"{prob_pos*100:.1f}%")

    nivel = int(prob_pos * 100)
    color = "#DC2626" if nivel >= 50 else "#16A34A" if nivel < 30 else "#D97706"
    st.markdown(f"""
    <div style="background:#E5E7EB;border-radius:8px;height:22px;width:100%;overflow:hidden;">
        <div style="background:{color};width:{nivel}%;height:100%;border-radius:8px;
                    display:flex;align-items:center;justify-content:flex-end;padding-right:8px;">
            <span style="color:white;font-size:12px;font-weight:600;">{nivel}%</span>
        </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("📋 Ver datos ingresados"):
        st.dataframe(datos.T.rename(columns={0: "Valor"}), use_container_width=True)

st.markdown("""
<div style="font-size:0.78rem;color:#9CA3AF;text-align:center;margin-top:2rem;">
    ⚠️ Esta herramienta es educativa y no sustituye el diagnóstico médico profesional.
</div>""", unsafe_allow_html=True)
