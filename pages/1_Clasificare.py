import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import shap

st.set_page_config(page_title="Clasificare - Diabet", page_icon="🩺", layout="wide")

@st.cache_resource
def load_data():
    return joblib.load('export_clasificare.pkl')


try:
    data = load_data()
    modele = data['modele']
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
except FileNotFoundError:
    st.error(
        "⚠️ Fișierul 'export_clasificare.pkl' nu a fost găsit. Asigură-te că ai rulat ultima celulă din notebook-ul de clasificare!")
    st.stop()

st.title("🩺 Clasificare: Predicția Riscului de Diabet")

st.markdown("""
### Descrierea problemei
În această secțiune, abordăm o problemă clasică de predicție medicală: identificarea riscului ca un pacient să aibă diabet, pe baza unor analize și măsurători de bază (glicemie, tensiune, BMI, vârstă etc.). 
Scopul este de a asista decizia medicală, modelul oferind un scor de risc (Clasa 0 - Sănătos, Clasa 1 - Diabet).
""")
st.sidebar.header("⚙️ Setări Model")
model_ales_nume = st.sidebar.selectbox("Alege modelul pentru analiză:", list(modele.keys()))
model = modele[model_ales_nume]
with st.expander(f"Vezi hiperparametrii modelului: {model_ales_nume}"):
    try:
        st.write(model.get_params())
    except AttributeError:
        parametri_siguri = {k: str(v) for k, v in vars(model).items() if not k.endswith('_')}
        st.write(parametri_siguri)

st.divider()
st.subheader("🔮 Fă o predicție pentru un pacient nou")
st.write(
    "Modifică valorile de mai jos (caracteristicile pacientului) pentru a vedea cum se schimbă predicția modelului selectat:")
col1, col2, col3, col4 = st.columns(4)
input_data = {}
for i, col in enumerate(X_train.columns):
    min_val = float(X_train[col].min())
    max_val = float(X_train[col].max())
    mean_val = float(X_train[col].mean())

    with [col1, col2, col3, col4][i % 4]:
        input_data[col] = st.slider(col, min_value=min_val, max_value=max_val, value=mean_val)
df_input = pd.DataFrame([input_data])
if st.button("Calculează Predicția", type="primary"):
    pred_class = model.predict(df_input)[0]
    try:
        pred_proba = model.predict_proba(df_input)[0][1] * 100
    except Exception:
        pred_proba = None

    st.markdown("### Rezultat Predicție:")
    if pred_class == 1:
        st.error("🚨 **PACIENT CU RISC DE DIABET**")
    else:
        st.success("✅ **PACIENT SĂNĂTOS (Fără risc major detectat)**")

    if pred_proba is not None:
        st.write(f"*Probabilitatea estimată pentru diabet este de: **{pred_proba:.2f}%***")

st.divider()
st.subheader(f"📊 Performanța modelului: {model_ales_nume} (pe datele de test)")
y_pred = model.predict(X_test)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Acuratețe", f"{accuracy_score(y_test, y_pred):.4f}")
m2.metric("Precizie", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
m3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
m4.metric("Scor F1", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Matricea de Confuzie**")
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Sănătos (0)", "Diabet (1)"])
    disp.plot(cmap="Blues", ax=ax_cm, colorbar=False)
    st.pyplot(fig_cm)

with col_g2:
    st.markdown("**Curba de Învățare**")
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )
        fig_lc, ax_lc = plt.subplots(figsize=(4, 3))
        ax_lc.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="red", label="Train")
        ax_lc.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color="green", label="Valid")
        ax_lc.set_xlabel("Exemple")
        ax_lc.set_ylabel("Scor")
        ax_lc.legend(loc="lower right", fontsize='small')
        ax_lc.grid(True)
        st.pyplot(fig_lc)
    except Exception:
        st.info("Curba de învățare nu poate fi generată pentru acest model cu setările actuale.")

st.divider()
st.subheader("🔍 Explicabilitatea Predicției (Analiza SHAP)")
with st.expander("Afișează analizele SHAP pentru modelul selectat"):
    st.write("Analiza SHAP ne arată cum influențează fiecare variabilă rezultatul final.")

    try:
        X_summary = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(model.predict, X_summary)
        shap_values_local = explainer.shap_values(df_input)

        col_shap1, col_shap2 = st.columns(2)

        with col_shap1:
            st.markdown("**Explicație Locală (Force Plot)**")
            fig_force = shap.force_plot(
                explainer.expected_value,
                shap_values_local[0],
                df_input.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()

        with col_shap2:
            st.markdown("**Importanța Caracteristicilor (Bar Plot)**")
            fig_bar, ax_bar = plt.subplots()
            shap.summary_plot(explainer.shap_values(X_test.iloc[:50]), X_test.iloc[:50], plot_type="bar", show=False)
            st.pyplot(fig_bar)
            plt.clf()

    except Exception as e:
        st.warning(
            "Analiza SHAP detaliată este disponibilă momentan doar pentru modelele bazate pe arbori în notebook. Aici afișăm importanța nativă a modelului dacă este disponibilă.")
        if hasattr(model, 'feature_importances_'):
            st.bar_chart(pd.Series(model.feature_importances_, index=X_train.columns))