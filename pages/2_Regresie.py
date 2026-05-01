import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import shap
st.set_page_config(page_title="Regresie - Imobiliare", page_icon="🏠", layout="wide")

@st.cache_resource
def load_data_reg():
    return joblib.load('export_regresie.pkl')

try:
    data_reg = load_data_reg()
    modele_reg = data_reg['modele']
    X_train_reg = data_reg['X_train']
    y_train_reg = data_reg['y_train']
    X_test_reg = data_reg['X_test']
    y_test_reg = data_reg['y_test']
except FileNotFoundError:
    st.error("⚠️ Fișierul 'export_regresie.pkl' nu a fost găsit!")
    st.stop()

st.title("🏠 Regresie: Estimarea Prețurilor Imobiliare")
st.markdown("""
### Descrierea problemei
Această secțiune vizează predicția prețului de vânzare al apartamentelor bazată pe caracteristici tehnice și locație. 
Variabila de ieșire este **Prețul (Euro)**, iar variabilele de intrare includ suprafața, numărul de camere și distanța față de centrul orașului.
""")
st.sidebar.header("⚙️ Setări Model")
model_nume = st.sidebar.selectbox("Alege modelul de regresie:", list(modele_reg.keys()))
model_reg = modele_reg[model_nume]

with st.expander("Vezi hiperparametrii modelului"):
    try:
        st.write(model_reg.get_params())
    except:
        st.write({k: str(v) for k, v in vars(model_reg).items() if not k.endswith('_')})

st.divider()
st.subheader("💰 Estimează prețul unui apartament")
col1, col2, col3 = st.columns(3)
input_reg = {}

with col1:
    input_reg['Suprafata_mp'] = st.number_input("Suprafață (mp)",
                                               min_value=float(X_train_reg['Suprafata_mp'].min()),
                                               max_value=float(X_train_reg['Suprafata_mp'].max()),
                                               value=float(X_train_reg['Suprafata_mp'].mean()))
with col2:
    input_reg['Nr_Camere'] = st.slider("Număr Camere", 1, 5, 2)
with col3:
    input_reg['Distanta_Centru_km'] = st.slider("Distanță Centru (km)", 0.5, 20.0, 5.0)
for col in X_train_reg.columns:
    if col not in input_reg:
        input_reg[col] = float(X_train_reg[col].mean())

df_input_reg = pd.DataFrame([input_reg])[X_train_reg.columns]

if st.button("Estimează Prețul", type="primary"):
    pret_predus = model_reg.predict(df_input_reg)[0]
    st.metric("Preț Estimat", f"{pret_predus:,.2f} EUR")

st.divider()
st.subheader(f"📊 Evaluare: {model_nume}")
y_pred_reg = model_reg.predict(X_test_reg)

c1, c2, c3 = st.columns(3)
c1.metric("R2 Score", f"{r2_score(y_test_reg, y_pred_reg):.4f}")
c2.metric("MAE (Euro)", f"{mean_absolute_error(y_test_reg, y_pred_reg):,.2f}")
c3.metric("RMSE (Euro)", f"{np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):,.2f}")

col_chart1, col_chart2 = st.columns(2)
with col_chart1:
    st.markdown("**Predicții vs Valori Reale**")
    fig_reg, ax_reg = plt.subplots(figsize=(5, 4))
    ax_reg.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    ax_reg.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--')
    ax_reg.set_xlabel("Real")
    ax_reg.set_ylabel("Predicție")
    st.pyplot(fig_reg)

with col_chart2:
    st.markdown("**Curba de Învățare**")
    try:
        train_sizes, train_scores, test_scores = learning_curve(model_reg, X_train_reg, y_train_reg, cv=3)
        fig_lc2, ax_lc2 = plt.subplots(figsize=(5, 4))
        ax_lc2.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
        ax_lc2.plot(train_sizes, np.mean(test_scores, axis=1), label="Valid")
        ax_lc2.legend()
        st.pyplot(fig_lc2)
    except:
        st.info("Curba nu este disponibilă pentru acest model.")
st.divider()
st.subheader("🔍 Explicabilitatea Prețului (SHAP)")
with st.expander("Vezi impactul caracteristicilor asupra prețului"):
    try:
        explainer_reg = shap.Explainer(model_reg.predict, shap.sample(X_train_reg, 50))
        shap_values_reg = explainer_reg(df_input_reg)
        fig_shap_reg = shap.plots.bar(shap_values_reg[0], show=False)
        st.pyplot(plt.gcf())
    except:
        st.write("Analiza SHAP detaliată este disponibilă în Notebook.")