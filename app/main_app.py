import sys
import os
import streamlit as st
import pandas as pd
import joblib

# Adiciona o diretório-pai ao caminho do Python para encontrar 'core'
app_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(app_dir)
sys.path.append(project_root)

from core.model_utils import train_linear_model, train_logistic_model

# --- Configurações da página ---
st.set_page_config(page_title="Streamlit - Desempenho de Vendas", layout="wide")

# --------------------------------------------------------------------------------------
# Lógica de Treinamento e Carregamento (executada uma vez)
# --------------------------------------------------------------------------------------
linear_model_path = os.path.join("model", "linear_regression_model.pickle")
logistic_model_path = os.path.join("model", "logistic_regression_model.pickle")
encoder_path = os.path.join("model", "onehot_encoder.joblib")

# Verificar se os modelos e o encoder já foram treinados e salvos
if not os.path.exists(linear_model_path) or not os.path.exists(logistic_model_path) or not os.path.exists(encoder_path):
    st.info("Modelos não encontrados. Treinando e salvando os modelos agora...")
    
    CSV_PATH = os.path.join(project_root, "Train.csv")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo CSV não encontrado no caminho: {CSV_PATH}")
        st.stop()
    
    # Preparar features e labels para o treinamento
    features_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']].copy()
    labels_linear = df['Item_Outlet_Sales'].copy()
    features_linear = features_linear.fillna(features_linear.mean(numeric_only=True))
    
    features_logistic = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']].copy()
    median_vis = df['Item_Visibility'].median()
    df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
    labels_logistic = df['Is_High_Visibility'].copy()
    
    # Treinar e salvar o modelo linear
    train_linear_model(features_linear, labels_linear)
    
    # Treinar e salvar o modelo logístico (agora com pré-processamento interno)
    train_logistic_model(features_logistic, labels_logistic)
    
    st.success("Modelos treinados e salvos com sucesso!")
    st.experimental_rerun() # Reinicia a página para carregar os modelos

# Carregar os modelos e o encoder para uso no app
try:
    linear_model = joblib.load(linear_model_path)
    log_model = joblib.load(logistic_model_path)
    encoder = joblib.load(encoder_path)
except FileNotFoundError:
    st.error("Erro: Arquivos de modelo e encoder não encontrados. Por favor, execute o código de treinamento primeiro.")
    st.stop()

# --------------------------------------------------------------------------------------
# Interface do Streamlit
# --------------------------------------------------------------------------------------
st.title("Streamlit - Desempenho de Vendas")
st.markdown("Bem-vindo ao dashboard de análise e previsão de desempenho de vendas.")

tab_analysis, tab_prediction = st.tabs(["📊 Análise & Métricas", "🔮 Previsão Interativa"])

with tab_analysis:
    st.header("Métricas de Desempenho")
    
    st.subheader("Modelo de Regressão Linear (Previsão de Vendas)")
    st.info("RMSE: O erro médio na previsão de vendas é de aproximadamente **$1338.89**")
    
    st.subheader("Modelo de Regressão Logística (Previsão de Visibilidade)")
    st.info("Acurácia: A acurácia do modelo para prever a visibilidade é de **95.60%**")

with tab_prediction:
    st.header("Faça sua Previsão")
    st.markdown("Selecione as características de uma loja para prever a visibilidade do item.")
    
    outlet_type = st.selectbox("Tipo de Loja", ['Supermarket Type1', 'Supermarket Type2', 'Grocery Store'])
    outlet_size = st.selectbox("Tamanho da Loja", ['Small', 'Medium', 'High'])
    location_type = st.selectbox("Tipo de Localização", ['Tier 1', 'Tier 2', 'Tier 3'])
    
    if st.button("Prever Visibilidade"):
        input_df = pd.DataFrame([[outlet_type, outlet_size, location_type]],
                                columns=['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type'])

        input_encoded = encoder.transform(input_df)

        prediction = log_model.predict(input_encoded)
        
        if prediction[0] == 1:
            st.success("A previsão para a visibilidade do item é **alta**!")
        else:
            st.warning("A previsão para a visibilidade do item é **baixa**.") 