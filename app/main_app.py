import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
from openai import OpenAI
import json
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Configuração inicial
# --------------------------------------------------
st.set_page_config(page_title="IA de Desempenho de Vendas", layout="wide")
st.title("📊 IA de Desempenho de Vendas — CSV apenas")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

MODEL_DIR = os.path.join(project_root, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pickle")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pickle")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.joblib")

# --------------------------------------------------
# OpenAI
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("📁 Importar Dados e Treinar")
uploaded_file = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])
retrain_btn = st.sidebar.button("🔁 Treinar modelo")

st.sidebar.header("🤖 Configuração GPT")
gpt_model_choice = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o"], index=0)
context_max_chars = st.sidebar.slider("Limite de contexto (caracteres)", 500, 20000, 4000, step=500)

train_status = st.sidebar.empty()

# --------------------------------------------------
# Importar funções do core
# --------------------------------------------------
from core.model_utils import train_linear_model, train_logistic_model

def load_or_train_models(csv_path=None, retrain=False):
    if not csv_path:
        raise ValueError("Nenhum arquivo CSV enviado.")

    df = pd.read_csv(csv_path)

    # Verificar se a coluna de vendas existe
    sales_col = None
    for col in df.columns:
        if "sales" in col.lower() or "vendas" in col.lower():
            sales_col = col
            break
    if not sales_col:
        raise ValueError("Nenhuma coluna relacionada a 'Sales' ou 'Vendas' foi encontrada.")

    # Tratar valores ausentes
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="mean")
        df[num_cols] = imputer_num.fit_transform(df[num_cols])

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # Modelos
    if retrain or not (
        os.path.exists(LINEAR_MODEL_PATH)
        and os.path.exists(LOGISTIC_MODEL_PATH)
        and os.path.exists(ENCODER_PATH)
    ):
        # Regressão Linear
        df_linear_features = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']]
        df_linear_labels = df[sales_col]
        train_linear_model(df_linear_features, df_linear_labels)

        # Regressão Logística
        median_vis = df['Item_Visibility'].median()
        df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
        df_logistic_features = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
        df_logistic_labels = df['Is_High_Visibility']
        train_logistic_model(df_logistic_features, df_logistic_labels)

    with open(LINEAR_MODEL_PATH, "rb") as f:
        linear_model = pickle.load(f)
    with open(LOGISTIC_MODEL_PATH, "rb") as f:
        logistic_model = pickle.load(f)
    encoder = joblib.load(ENCODER_PATH)
    return linear_model, logistic_model, encoder, df, sales_col

# --------------------------------------------------
# Calcular métricas
# --------------------------------------------------
def calculate_metrics(linear_model, logistic_model, encoder, df, sales_col):
    X_lin = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']]
    y_lin = df[sales_col]
    y_pred_lin = linear_model.predict(X_lin)
    rmse = np.sqrt(mean_squared_error(y_lin, y_pred_lin))

    median_vis = df['Item_Visibility'].median()
    df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
    X_log = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
    X_log_enc = encoder.transform(X_log)
    y_true_log = df['Is_High_Visibility']
    y_pred_log = logistic_model.predict(X_log_enc)

    acc = accuracy_score(y_true_log, y_pred_log)
    cm = confusion_matrix(y_true_log, y_pred_log)
    cls_report = classification_report(y_true_log, y_pred_log, output_dict=True, zero_division=0)

    tp, fn = cm[1, 1], cm[1, 0]
    fp, tn = cm[0, 1], cm[0, 0]
    odds_ratio = (tp * tn) / ((fp * fn) + 1e-9)

    return {
        "rmse": rmse,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "odds_ratio": odds_ratio,
        "y_true_linear": y_lin,
        "y_pred_linear": y_pred_lin,
    }

# --------------------------------------------------
# Execução
# --------------------------------------------------
linear_model = logistic_model = encoder = metrics = None

if retrain_btn:
    if not uploaded_file:
        train_status.error("❌ Nenhum arquivo CSV enviado.")
    else:
        save_path = os.path.join(MODEL_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        train_status.info("⚙️ Treinando modelos...")
        try:
            linear_model, logistic_model, encoder, df, sales_col = load_or_train_models(save_path, retrain=True)
            metrics = calculate_metrics(linear_model, logistic_model, encoder, df, sales_col)
            train_status.success("✅ Modelos treinados com sucesso!")
        except Exception as e:
            train_status.error(f"Erro ao treinar o modelo: {e}")

# --------------------------------------------------
# Exibir métricas
# --------------------------------------------------
if metrics:
    st.markdown("## ✅ Métricas do Modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['rmse']:.2f}")
    c2.metric("Acurácia", f"{metrics['accuracy']*100:.2f}%")
    c3.metric("Odds Ratio", f"{metrics['odds_ratio']:.2f}")

    with st.expander("📊 Detalhes do Modelo"):
        st.write("Matriz de Confusão:")
        st.dataframe(pd.DataFrame(metrics['confusion_matrix'], columns=["Prev:0","Prev:1"], index=["Real:0","Real:1"]))

        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(metrics['classification_report']).T)

    st.markdown("### 🔍 Visualização")
    comp_df = pd.DataFrame({
        "Vendas Reais": metrics["y_true_linear"],
        "Vendas Previstas": metrics["y_pred_linear"]
    })
    fig_scatter = px.scatter(comp_df, x="Vendas Reais", y="Vendas Previstas", title="Vendas Reais vs Previstas")
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.info("Envie um arquivo CSV e clique em **Treinar modelo** para começar.")

# --------------------------------------------------
# Chat GPT
# --------------------------------------------------
st.markdown("---")
st.header("💬 Chat Inteligente")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_msg = st.chat_input("Escreva sua pergunta (Enter para enviar):")

def call_gpt(question, metrics):
    if not OPENAI_API_KEY:
        return "❌ Chave OpenAI ausente."
    client = OpenAI(api_key=OPENAI_API_KEY)

    ctx = ""
    if metrics:
        ctx = json.dumps({
            "RMSE": metrics["rmse"],
            "Acurácia": metrics["accuracy"],
            "Odds Ratio": metrics["odds_ratio"]
        }, indent=2)
    if len(ctx) > context_max_chars:
        ctx = ctx[:context_max_chars] + "\n... (truncado)"

    messages = [
        {"role": "system", "content": "Você é um analista de dados especialista em desempenho de vendas."},
        {"role": "user", "content": f"Contexto:\n{ctx}\n\nPergunta: {question}"}
    ]
    try:
        response = client.chat.completions.create(
            model=gpt_model_choice,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erro: {e}"

if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    answer = call_gpt(user_msg, metrics)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()
