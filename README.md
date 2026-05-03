# Analiza Comparată a Modelelor de Machine Learning în Regresie și Clasificare

Acest proiect simulează fluxul complet de lucru al unei probleme reale de Machine Learning, acoperind etapele de la înțelegerea datelor până la implementarea unei aplicații interactive.

## 📂 Structura Proiectului
* **`notebook_clasificare.ipynb`**: Documentația tehnică pentru problema de clasificare (predicția riscului de diabet).
* **`notebook_regresie.ipynb`**: Documentația tehnică pentru problema de regresie (estimarea prețurilor imobiliare).
* **`app.py`**: Fișierul principal al aplicației Streamlit.
* **`pages/`**: Pagini interactive dedicate pentru Clasificare și Regresie
* **`export_*.pkl`**: Modelele antrenate și optimizate, salvate pentru utilizare rapidă.
* **`date_*.csv`**: Seturile de date utilizate în analiză.

## 🎯 Obiectivele Proiectului
* Realizarea unei analize exploratorii a datelor (EDA) detaliate.
* Antrenarea și compararea performanței mai multor algoritmi (XGBoost, CatBoost, Logistic Regression etc.).
* Optimizarea modelelor prin ajustarea hiperparametrilor.
* Interpretarea deciziilor modelelor folosind analiza **SHAP** (Explicabilitate locală și globală).
* Evaluarea capacității de generalizare prin curbele de învățare.

## 🚀 Cum se rulează aplicația
Pentru a vizualiza dashboard-ul interactiv, asigurați-vă că aveți instalate bibliotecile necesare, apoi rulați următoarea comandă în terminal:
```bash
streamlit run app.py