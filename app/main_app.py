import sys
import os
import subprocess

# --------------------------------------------------------------------------------------
# Wrapper para rodar tanto com `python app.py` quanto com `streamlit run app.py`
# --------------------------------------------------------------------------------------
if __name__ == "__main__" and not os.environ.get("STREAMLIT_RUN"):
    script_path = os.path.abspath(__file__)
    print(f"🔄 Iniciando o Streamlit para {script_path}...")
    os.environ["STREAMLIT_RUN"] = "1"  # evita loop infinito
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
    sys.exit(0)

# --------------------------------------------------------------------------------------
# A partir daqui é o código normal do Streamlit
# --------------------------------------------------------------------------------------
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
import streamlit as st

# Configuração de diretórios
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
sys.path.append(project_root)

from core.model_utils import train_linear_model, train_logistic_model

linear_model_path = os.path.join("model", "linear_regression_model.pickle")
logistic_model_path = os.path.join("model", "logistic_regression_model.pickle")
encoder_path = os.path.join("model", "onehot_encoder.joblib")

CSV_PATH = os.path.join(project_root, "Train.csv")

# --------------------------------------------------------------------------------------
# Funções auxiliares
# --------------------------------------------------------------------------------------
def load_or_train_models():
    if not os.path.exists(linear_model_path) or not os.path.exists(logistic_model_path) or not os.path.exists(encoder_path):
        st.info("Modelos não encontrados. Treinando e salvando agora...")

        df = pd.read_csv(CSV_PATH)

        features_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].copy()
        labels_linear = df['Item_Outlet_Sales'].copy()
        features_linear = features_linear.fillna(features_linear.mean(numeric_only=True))

        features_logistic = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']].copy()
        median_vis = df['Item_Visibility'].median()
        df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
        labels_logistic = df['Is_High_Visibility'].copy()

        train_linear_model(features_linear, labels_linear)
        train_logistic_model(features_logistic, labels_logistic)

        st.success("✅ Modelos treinados e salvos com sucesso!")
        st.experimental_rerun()

    # --- carregar os modelos salvos ---
    import pickle
    with open(linear_model_path, "rb") as f:
        linear_model = pickle.load(f)

    with open(logistic_model_path, "rb") as f:
        log_model = pickle.load(f)

    encoder = joblib.load(encoder_path)

    return linear_model, log_model, encoder



def calcular_metricas(df, linear_model, log_model, encoder):
    if "Is_High_Visibility" not in df.columns:
        median_vis = df['Item_Visibility'].median()
        df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)

    y_true_linear = df['Item_Outlet_Sales']
    X_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].fillna(df.mean(numeric_only=True))
    y_pred_linear = linear_model.predict(X_linear)
    rmse = np.sqrt(mean_squared_error(y_true_linear, y_pred_linear))

    y_true_log = df['Is_High_Visibility']
    X_log = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
    X_log_encoded = encoder.transform(X_log)
    y_pred_log = log_model.predict(X_log_encoded)
    acc = accuracy_score(y_true_log, y_pred_log) * 100

    return rmse, acc

# --------------------------------------------------------------------------------------
# Interface Streamlit
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Streamlit - Desempenho de Vendas", layout="wide")
st.title("Streamlit - Desempenho de Vendas")
st.markdown("Bem-vindo ao dashboard de análise e previsão de desempenho de vendas.")

linear_model, log_model, encoder = load_or_train_models()
df = pd.read_csv(CSV_PATH)

tab_analysis, tab_prediction = st.tabs(["📊 Análise & Métricas", "🔮 Previsão Interativa"])

# --- Aba de Análise ---
with tab_analysis:
    st.header("Métricas de Desempenho")
    rmse, acc = calcular_metricas(df, linear_model, log_model, encoder)

    st.subheader("Modelo de Regressão Linear (Previsão de Vendas)")
    st.info(f"RMSE: O erro médio na previsão de vendas é de aproximadamente **${rmse:.2f}**")

    st.subheader("Modelo de Regressão Logística (Previsão de Visibilidade)")
    st.info(f"Acurácia: A acurácia do modelo para prever a visibilidade é de **{acc:.2f}%**")

# --- Aba de Previsão ---
with tab_prediction:
    st.header("Faça sua Previsão")

    outlet_types = sorted(df['Outlet_Type'].dropna().unique())
    outlet_sizes = sorted(df['Outlet_Size'].dropna().unique())
    location_types = sorted(df['Outlet_Location_Type'].dropna().unique())

    st.markdown("### 🔮 Previsão de Visibilidade (Classificação)")
    outlet_type = st.selectbox("Tipo de Loja", outlet_types)
    outlet_size = st.selectbox("Tamanho da Loja", outlet_sizes)
    location_type = st.selectbox("Tipo de Localização", location_types)

    if st.button("Prever Visibilidade"):
        input_df = pd.DataFrame([[outlet_type, outlet_size, location_type]],
                                columns=['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type'])
        input_encoded = encoder.transform(input_df)
        prediction = log_model.predict(input_encoded)
        if prediction[0] == 1:
            st.success("✅ A previsão para a visibilidade do item é **alta**!")
        else:
            st.warning("⚠️ A previsão para a visibilidade do item é **baixa**.")

    st.markdown("---")

    st.markdown("### 📈 Previsão de Vendas (Regressão Linear)")
    item_weight = st.number_input("Peso do Item", min_value=0.0, step=0.1)
    item_visibility = st.number_input("Visibilidade do Item", min_value=0.0, step=0.01)
    item_mrp = st.number_input("Preço Máximo de Varejo (MRP)", min_value=0.0, step=0.1)

    if st.button("Prever Vendas"):
        input_df = pd.DataFrame([[item_weight, item_visibility, item_mrp]],
                                columns=['Item_Weight', 'Item_Visibility', 'Item_MRP'])
        prediction = linear_model.predict(input_df)
        st.success(f"💰 A previsão de vendas para esse item é aproximadamente **${prediction[0]:.2f}**")
