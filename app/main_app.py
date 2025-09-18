import os
import sys
import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sqlalchemy import create_engine, inspect

# --------------------------
# Garantir que o Streamlit encontre o pacote core
# --------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.model_utils import train_linear_model, train_logistic_model
from core.db_utils import execute_sql_file  # função para executar arquivos .sql

# --------------------------
# Configuração de caminhos
# --------------------------
MODEL_DIR = os.path.join(project_root, "model")
DATA_DIR = os.path.join(project_root, "core", "data")
CSV_PATH = os.path.join(project_root, "Train.csv")

LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pickle")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pickle")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.joblib")

# --------------------------
# Configuração do banco SQLite
# --------------------------
DB_PATH = os.path.join(project_root, "BigMarkSales.db")
ENGINE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(ENGINE_URL, echo=False)

# --------------------------
# Funções auxiliares
# --------------------------
def ensure_tables_and_populate():
    """Cria tabelas se não existirem e popula com dados do CSV"""
    inspector = inspect(engine)
    sql_files = {
        "spec_features_linear": os.path.join(DATA_DIR, "spec_features_linear.sql"),
        "spec_features_logistic": os.path.join(DATA_DIR, "spec_features_logistic.sql"),
        "spec_labels_linear": os.path.join(DATA_DIR, "spec_labels_linear.sql"),
        "spec_labels_logistic": os.path.join(DATA_DIR, "spec_labels_logistic.sql"),
    }

    df_csv = pd.read_csv(CSV_PATH)

    # Criação das tabelas e inserção de dados
    for table_name, sql_file in sql_files.items():
        if not inspector.has_table(table_name):
            # Cria a tabela vazia a partir do .sql
            execute_sql_file(engine, sql_file)
            st.info(f"Tabela {table_name} criada a partir de {sql_file}")

        # Inserir dados do CSV
        if table_name == "spec_features_linear":
            df_insert = df_csv[['Item_Weight', 'Item_Visibility', 'Item_MRP']].fillna(df_csv.mean(numeric_only=True))
        elif table_name == "spec_features_logistic":
            df_insert = df_csv[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']].fillna("Unknown")
        elif table_name == "spec_labels_linear":
            df_insert = df_csv[['Item_Outlet_Sales']]
        elif table_name == "spec_labels_logistic":
            median_vis = df_csv['Item_Visibility'].median()
            df_insert = pd.DataFrame({'Is_High_Visibility': (df_csv['Item_Visibility'] > median_vis).astype(int)})

        # Inserir dados no banco
        df_insert.to_sql(table_name, con=engine, if_exists="replace", index=False)
        st.info(f"Tabela {table_name} populada com {len(df_insert)} linhas")



def load_table(table_name):
    """Carrega uma tabela SQL para um DataFrame"""
    return pd.read_sql_table(table_name, engine)


def load_or_train_models(train=False):
    """Carrega ou treina os modelos e garante que tabelas existam"""
    ensure_tables_and_populate()


    # Se não existir algum modelo ou solicitar treino
    if train or not (os.path.exists(LINEAR_MODEL_PATH) and os.path.exists(LOGISTIC_MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        st.info("Treinando modelos...")

        # Carregar os dados originais
        df = pd.read_csv(CSV_PATH)

        # Features e labels Linear
        df_linear_features = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].fillna(df.mean(numeric_only=True))
        df_linear_labels = df['Item_Outlet_Sales']

        # Features e labels Logistic
        median_vis = df['Item_Visibility'].median()
        df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
        df_logistic_features = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
        df_logistic_labels = df['Is_High_Visibility']

        # Treinar modelos
        train_linear_model(df_linear_features, df_linear_labels)
        train_logistic_model(df_logistic_features, df_logistic_labels)

        st.success("✅ Modelos treinados e salvos!")

    # Carregar modelos salvos
    with open(LINEAR_MODEL_PATH, "rb") as f:
        linear_model = pickle.load(f)
    with open(LOGISTIC_MODEL_PATH, "rb") as f:
        log_model = pickle.load(f)
    encoder = joblib.load(ENCODER_PATH)

    # Carregar tabelas SQL
    df_features_linear = load_table("spec_features_linear")
    df_features_logistic = load_table("spec_features_logistic")

    return linear_model, log_model, encoder, df_features_linear, df_features_logistic


def calcular_metricas(linear_model, log_model, encoder):
    df = pd.read_csv(CSV_PATH)

    # Regressão Linear
    X_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].fillna(df.mean(numeric_only=True))
    y_true_linear = df['Item_Outlet_Sales']
    y_pred_linear = linear_model.predict(X_linear)
    rmse = np.sqrt(mean_squared_error(y_true_linear, y_pred_linear))

    # Regressão Logística
    median_vis = df['Item_Visibility'].median()
    df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
    X_log = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
    X_log_encoded = encoder.transform(X_log)
    y_true_log = df['Is_High_Visibility']
    y_pred_log = log_model.predict(X_log_encoded)
    acc = accuracy_score(y_true_log, y_pred_log) * 100

    return rmse, acc


# --------------------------
# Interface Streamlit
# --------------------------
st.set_page_config(page_title="Desempenho de Vendas", layout="wide")
st.title("Streamlit - Desempenho de Vendas")
st.markdown("Dashboard para análise e previsão de desempenho de vendas.")

# Treinar modelos
if st.button("Treinar Modelos"):
    linear_model, log_model, encoder, df_linear, df_logistic = load_or_train_models(train=True)
else:
    linear_model, log_model, encoder, df_linear, df_logistic = load_or_train_models(train=False)

# --------------------------
# Abas do Streamlit
# --------------------------
tab_analysis, tab_prediction, tab_tables = st.tabs(["📊 Métricas", "🔮 Previsão", "📋 Tabelas"])

# Métricas
with tab_analysis:
    st.header("Métricas de Desempenho")
    rmse, acc = calcular_metricas(linear_model, log_model, encoder)
    st.subheader("Regressão Linear")
    st.info(f"RMSE: {rmse:.2f}")
    st.subheader("Regressão Logística")
    st.info(f"Acurácia: {acc:.2f}%")

# Previsão interativa
with tab_prediction:
    st.header("Previsão Interativa")

    outlet_types = sorted(df_logistic['Outlet_Type'].dropna().unique())
    outlet_sizes = sorted(df_logistic['Outlet_Size'].dropna().unique())
    location_types = sorted(df_logistic['Outlet_Location_Type'].dropna().unique())

    st.markdown("### 🔮 Previsão de Visibilidade")
    outlet_type = st.selectbox("Tipo de Loja", outlet_types)
    outlet_size = st.selectbox("Tamanho da Loja", outlet_sizes)
    location_type = st.selectbox("Tipo de Localização", location_types)

    if st.button("Prever Visibilidade"):
        input_df = pd.DataFrame([[outlet_type, outlet_size, location_type]],
                                columns=['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type'])
        input_encoded = encoder.transform(input_df)
        prediction = log_model.predict(input_encoded)
        if prediction[0] == 1:
            st.success("✅ Visibilidade alta")
        else:
            st.warning("⚠️ Visibilidade baixa")

    st.markdown("---")
    st.markdown("### 📈 Previsão de Vendas")
    item_weight = st.number_input("Peso do Item", min_value=0.0, step=0.1)
    item_visibility = st.number_input("Visibilidade do Item", min_value=0.0, step=0.01)
    item_mrp = st.number_input("Preço Máximo de Varejo", min_value=0.0, step=0.1)

    if st.button("Prever Vendas"):
        input_df = pd.DataFrame([[item_weight, item_visibility, item_mrp]],
                                columns=['Item_Weight', 'Item_Visibility', 'Item_MRP'])
        prediction = linear_model.predict(input_df)
        st.success(f"💰 Previsão de vendas: ${prediction[0]:.2f}")

# Tabelas
with tab_tables:
    st.header("Verificação das Tabelas SQL")
    tabelas = ["spec_features_linear", "spec_features_logistic", "spec_labels_linear", "spec_labels_logistic"]
    for table_name in tabelas:
        try:
            df = pd.read_sql_table(table_name, engine)
            st.subheader(f"Tabela: {table_name}")
            if df.empty:
                st.warning("⚠️ A tabela está vazia!")
            else:
                st.dataframe(df.head(10))  # Mostra as 10 primeiras linhas
        except Exception as e:
            st.error(f"❌ Erro ao carregar a tabela {table_name}: {e}")
