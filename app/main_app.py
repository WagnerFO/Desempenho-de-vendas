import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import mean_squared_error, accuracy_score
from sqlalchemy import text

# --------------------------
# Garantir que o Streamlit encontre o pacote core
# --------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model_utils import train_linear_model, train_logistic_model
from core.setup_mysql import (
    create_database_if_not_exists,
    execute_sql_folder,
    get_engine,
    insert_dataframe
)

# --------------------------
# Caminhos dos modelos e CSV
# --------------------------
linear_model_path = os.path.join("model", "linear_regression_model.pickle")
logistic_model_path = os.path.join("model", "logistic_regression_model.pickle")
encoder_path = os.path.join("model", "onehot_encoder.joblib")
CSV_PATH = os.path.join(project_root, "Train.csv")

# =========================
# FUNÇÕES AUXILIARES
# =========================
def load_or_train_models(train=False):
    """Carrega modelos ou treina se train=True"""
    engine = get_engine()

    if train or not (
        os.path.exists(linear_model_path)
        and os.path.exists(logistic_model_path)
        and os.path.exists(encoder_path)
    ):
        st.info("Treinando modelos...")

        # 1) Garantir banco e tabelas
        create_database_if_not_exists()
        execute_sql_folder()

        # 2) Ler CSV e preparar features/labels
        df = pd.read_csv(CSV_PATH)

        # --- Linear ---
        features_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].fillna(df.mean(numeric_only=True))
        labels_linear = df[['Item_Outlet_Sales']]

        # --- Logístico ---
        features_logistic = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']].copy()
        median_vis = df['Item_Visibility'].median()
        df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
        labels_logistic = df[['Is_High_Visibility']].copy()

        # 3) Salvar no MySQL
        insert_dataframe(features_linear, "spec_features_linear")
        insert_dataframe(labels_linear, "spec_labels_linear")
        insert_dataframe(features_logistic, "spec_features_logistic")
        insert_dataframe(labels_logistic, "spec_labels_logistic")

        # 4) Ler do banco para treinar
        with engine.connect() as conn:
            features_linear = pd.read_sql(text("SELECT * FROM spec_features_linear"), conn)
            labels_linear = pd.read_sql(text("SELECT * FROM spec_labels_linear"), conn)
            features_logistic = pd.read_sql(text("SELECT * FROM spec_features_logistic"), conn)
            labels_logistic = pd.read_sql(text("SELECT * FROM spec_labels_logistic"), conn)

        # 5) Treinar
        train_linear_model(features_linear, labels_linear.values.ravel())
        train_logistic_model(features_logistic, labels_logistic.values.ravel())

        st.success("✅ Modelos treinados e salvos!")

    # Carregar modelos
    with open(linear_model_path, "rb") as f:
        linear_model = pickle.load(f)
    with open(logistic_model_path, "rb") as f:
        log_model = pickle.load(f)
    encoder = joblib.load(encoder_path)

    return linear_model, log_model, encoder


def calcular_metricas(linear_model, log_model, encoder):
    """Calcula métricas lendo dados direto do MySQL"""
    engine = get_engine()
    with engine.connect() as conn:
        features_linear = pd.read_sql(text("SELECT * FROM spec_features_linear"), conn)
        labels_linear = pd.read_sql(text("SELECT * FROM spec_labels_linear"), conn)
        features_logistic = pd.read_sql(text("SELECT * FROM spec_features_logistic"), conn)
        labels_logistic = pd.read_sql(text("SELECT * FROM spec_labels_logistic"), conn)

    # --- RMSE Linear ---
    y_true_linear = labels_linear.values.ravel()
    y_pred_linear = linear_model.predict(features_linear)
    rmse = np.sqrt(mean_squared_error(y_true_linear, y_pred_linear))

    # --- Acurácia Logística ---
    y_true_log = labels_logistic.values.ravel()
    X_log_encoded = encoder.transform(features_logistic)
    y_pred_log = log_model.predict(X_log_encoded)
    acc = accuracy_score(y_true_log, y_pred_log) * 100

    return rmse, acc, features_logistic


# =========================
# STREAMLIT
# =========================
st.set_page_config(page_title="Streamlit - Desempenho de Vendas", layout="wide")
st.title("Streamlit - Desempenho de Vendas")
st.markdown("Bem-vindo ao dashboard de análise e previsão de desempenho de vendas.")

# Botão de treino
if st.button("Treinar Modelos"):
    linear_model, log_model, encoder = load_or_train_models(train=True)
else:
    linear_model, log_model, encoder = load_or_train_models(train=False)

tab_analysis, tab_prediction = st.tabs(["📊 Análise & Métricas", "🔮 Previsão Interativa"])

# --- Aba Análise ---
with tab_analysis:
    st.header("Métricas de Desempenho")
    rmse, acc, features_logistic = calcular_metricas(linear_model, log_model, encoder)
    st.subheader("Modelo de Regressão Linear (Previsão de Vendas)")
    st.info(f"RMSE: ${rmse:.2f}")
    st.subheader("Modelo de Regressão Logística (Previsão de Visibilidade)")
    st.info(f"Acurácia: {acc:.2f}%")

# --- Aba Previsão ---
with tab_prediction:
    st.header("Faça sua Previsão")
    outlet_types = sorted(features_logistic['Outlet_Type'].dropna().unique())
    outlet_sizes = sorted(features_logistic['Outlet_Size'].dropna().unique())
    location_types = sorted(features_logistic['Outlet_Location_Type'].dropna().unique())

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
            st.success("✅ Visibilidade do item: alta")
        else:
            st.warning("⚠️ Visibilidade do item: baixa")

    st.markdown("---")
    st.markdown("### 📈 Previsão de Vendas (Regressão Linear)")
    item_weight = st.number_input("Peso do Item", min_value=0.0, step=0.1)
    item_visibility = st.number_input("Visibilidade do Item", min_value=0.0, step=0.01)
    item_mrp = st.number_input("Preço Máximo de Varejo (MRP)", min_value=0.0, step=0.1)

    if st.button("Prever Vendas"):
        input_df = pd.DataFrame([[item_weight, item_visibility, item_mrp]],
                                columns=['Item_Weight', 'Item_Visibility', 'Item_MRP'])
        prediction = linear_model.predict(input_df)
        st.success(f"💰 Previsão de vendas: ${prediction[0]:.2f}")
