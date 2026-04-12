import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Diabetes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
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
    .metric-label {
        font-size: 0.82rem;
        color: #6B7280;
        margin-bottom: 2px;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #374151;
    }
    .footer-note {
        font-size: 0.78rem;
        color: #9CA3AF;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Encabezado
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 Predictor de Diabetes</h1>
    <p style="color: #6B7280; font-size: 0.95rem;">
        Modelo entrenado con FLAML (AutoML) · Dataset Pima Indians Diabetes
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Cargar modelo
# ─────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    ruta = "modelo_diabetes_flaml.pkl"
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None

modelo = cargar_modelo()

if modelo is None:
    st.warning("""
    ⚠️ **Modelo no encontrado.**  
    Ejecuta el notebook de Colab, descarga `modelo_diabetes_flaml.pkl`
    y colócalo en la misma carpeta que este archivo `app.py`.
    
    Mientras tanto, puedes explorar la interfaz con valores de ejemplo.
    """)
    usar_demo = True
else:
    usar_demo = False
    st.success("✅ Modelo cargado correctamente")

# ─────────────────────────────────────────────
# Formulario de entrada
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Datos del paciente</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Número de embarazos",
        min_value=0, max_value=20, value=2, step=1,
        help="Número de veces que la paciente ha estado embarazada"
    )
    glucose = st.number_input(
        "Glucosa en plasma (mg/dL)",
        min_value=0, max_value=300, value=120, step=1,
        help="Concentración de glucosa a las 2 horas (prueba de tolerancia oral)"
    )
    blood_pressure = st.number_input(
        "Presión diastólica (mmHg)",
        min_value=0, max_value=150, value=72, step=1,
        help="Presión sanguínea diastólica"
    )
    skin_thickness = st.number_input(
        "Grosor del pliegue cutáneo (mm)",
        min_value=0, max_value=100, value=23, step=1,
        help="Grosor del pliegue cutáneo del tríceps"
    )

with col2:
    insulin = st.number_input(
        "Insulina sérica (μU/mL)",
        min_value=0, max_value=900, value=85, step=1,
        help="Insulina sérica a las 2 horas"
    )
    bmi = st.number_input(
        "IMC (kg/m²)",
        min_value=0.0, max_value=70.0, value=26.5, step=0.1,
        help="Índice de masa corporal: peso(kg) / altura(m)²"
    )
    dpf = st.number_input(
        "Función pedigrí de diabetes",
        min_value=0.000, max_value=3.000, value=0.351, step=0.001,
        format="%.3f",
        help="Puntuación que refleja el historial familiar de diabetes"
    )
    age = st.number_input(
        "Edad (años)",
        min_value=1, max_value=120, value=33, step=1,
        help="Edad de la paciente"
    )

# ─────────────────────────────────────────────
# Botón de predicción
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predecir = st.button("🔍 Generar predicción", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────
if predecir:
    datos = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    if usar_demo:
        # Predicción demo con lógica simple (sin modelo real)
        riesgo_demo = (
            (glucose > 140) * 0.4 +
            (bmi > 30) * 0.25 +
            (age > 45) * 0.2 +
            (dpf > 0.5) * 0.15
        )
        prob_positivo = min(max(riesgo_demo, 0.05), 0.95)
        prob_negativo = 1 - prob_positivo
        prediccion = 1 if prob_positivo >= 0.5 else 0
        st.info("ℹ️ Resultado generado en **modo demostración** (modelo no cargado).")
    else:
        prediccion = modelo.predict(datos)[0]
        probs = modelo.predict_proba(datos)[0]
        prob_negativo = probs[0]
        prob_positivo = probs[1]

    st.divider()
    st.markdown('<div class="section-title">📊 Resultado del análisis</div>', unsafe_allow_html=True)

    if prediccion == 1:
        st.markdown(f"""
        <div class="result-box-positive">
            <h3 style="color:#DC2626; margin:0 0 0.5rem 0;">⚠️ Alto riesgo de diabetes detectado</h3>
            <p style="margin:0; color:#7F1D1D; font-size:0.95rem;">
                El modelo indica que el paciente <strong>podría tener diabetes</strong>.
                Se recomienda consultar con un médico especialista.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-negative">
            <h3 style="color:#16A34A; margin:0 0 0.5rem 0;">✅ Bajo riesgo de diabetes</h3>
            <p style="margin:0; color:#14532D; font-size:0.95rem;">
                El modelo indica que el paciente <strong>probablemente no tiene diabetes</strong>.
                Se recomienda mantener hábitos saludables y revisiones periódicas.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Probabilidades
    st.markdown("**Probabilidades del modelo:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="Sin diabetes (clase 0)",
            value=f"{prob_negativo*100:.1f}%"
        )
    with col_b:
        st.metric(
            label="Con diabetes (clase 1)",
            value=f"{prob_positivo*100:.1f}%"
        )

    # Barra de probabilidad visual
    st.markdown("**Nivel de riesgo:**")
    nivel = int(prob_positivo * 100)
    color_barra = "#DC2626" if nivel >= 50 else "#16A34A" if nivel < 30 else "#D97706"
    st.markdown(f"""
    <div style="background:#E5E7EB; border-radius:8px; height:22px; width:100%; overflow:hidden;">
        <div style="background:{color_barra}; width:{nivel}%; height:100%; border-radius:8px;
                    display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
            <span style="color:white; font-size:12px; font-weight:600;">{nivel}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Resumen de datos ingresados
    with st.expander("📋 Ver datos ingresados"):
        st.dataframe(datos.T.rename(columns={0: "Valor"}), use_container_width=True)

# ─────────────────────────────────────────────
# Información del modelo (sidebar)
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Sobre el modelo")
    st.markdown("""
    **Dataset:** Pima Indians Diabetes  
    **Fuente:** UCI Machine Learning Repository  
    **Muestras:** 768 pacientes  
    **Clases:** No diabetes (0) / Diabetes (1)  
    
    ---
    **AutoML:** FLAML  
    **Métrica:** ROC-AUC  
    **Validación:** 5-fold CV  
    **Split:** 80% train / 20% test  
    
    ---
    **Variables de entrada:**
    - Pregnancies
    - Glucose ⭐
    - BloodPressure
    - SkinThickness
    - Insulin
    - BMI ⭐
    - DiabetesPedigreeFunction
    - Age ⭐
    
    *(⭐ = variables de mayor impacto)*
    """)
    
    st.markdown("""
    ---
    **⚠️ Aviso médico:**  
    Esta herramienta es únicamente de carácter educativo y no sustituye el diagnóstico médico profesional.
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-note">
    Desarrollado con FLAML + Streamlit · Proyecto AutoML
</div>
""", unsafe_allow_html=True)
