import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
from dotenv import load_dotenv
from openai import OpenAI
import json
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Configuração inicial
# ------------------------------
st.set_page_config(page_title="IA de Desempenho de Vendas", layout="wide")
st.title("📊 IA de Desempenho de Vendas — Automática")

project_root = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(project_root, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression_model.pickle")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pickle")
ENCODER_PATH = os.path.join(MODEL_DIR, "onehot_encoder.joblib")

# ------------------------------
# OpenAI
# ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("📁 Importar Dados e Treinar")
uploaded_file = st.sidebar.file_uploader("Envie um arquivo CSV", type=["csv"])
retrain_btn = st.sidebar.button("🔁 Treinar modelo")
st.sidebar.markdown("---")

st.sidebar.header("🤖 Configuração GPT")
gpt_model_choice = st.sidebar.selectbox("Modelo GPT", ["gpt-4o-mini", "gpt-4o"], index=0)
context_max_chars = st.sidebar.slider("Limite de contexto (caracteres)", 500, 20000, 4000, step=500)

train_status = st.sidebar.empty()

# ------------------------------
# Inicializa session_state
# ------------------------------
for key in ["chat_history","linear_model","logistic_model","encoder","metrics","df","X_linear","y_linear"]:
    if key not in st.session_state:
        st.session_state[key] = None

if "chat_history" not in st.session_state or st.session_state.chat_history is None:
    st.session_state.chat_history = []

# ------------------------------
# Funções de treino
# ------------------------------
def train_linear_model(X, y):
    lr = LinearRegression()
    lr.fit(X, y)
    with open(LINEAR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)
    return lr

def train_logistic_model(X, y):
    try:
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")
    X_encoded = encoder.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, encoder

def load_or_train_models(csv_path, retrain=False):
    df = pd.read_csv(csv_path)

    required_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 
                     'Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Item_Outlet_Sales']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    # Preenchimento de valores ausentes
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Item_Visibility'].fillna(df['Item_Visibility'].mean(), inplace=True)
    df['Item_MRP'].fillna(df['Item_MRP'].mean(), inplace=True)
    df['Outlet_Type'].fillna('Unknown', inplace=True)
    df['Outlet_Size'].fillna('Medium', inplace=True)
    df['Outlet_Location_Type'].fillna('Tier 2', inplace=True)

    # Linear
    X_linear = df[['Item_Weight', 'Item_Visibility', 'Item_MRP']]
    y_linear = df['Item_Outlet_Sales']

    # Logistic
    median_vis = df['Item_Visibility'].median()
    df['Is_High_Visibility'] = (df['Item_Visibility'] > median_vis).astype(int)
    X_log = df[['Outlet_Type', 'Outlet_Size', 'Outlet_Location_Type']]
    y_log = df['Is_High_Visibility']

    # Treinar ou carregar modelos
    linear_model = logistic_model = encoder = None
    if retrain or not (os.path.exists(LINEAR_MODEL_PATH) and os.path.exists(LOGISTIC_MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        train_status.info("⚙️ Treinando modelos...")
        linear_model = train_linear_model(X_linear, y_linear)
        logistic_model, encoder = train_logistic_model(X_log, y_log)
        train_status.success("✅ Modelos treinados com sucesso!")
    else:
        with open(LINEAR_MODEL_PATH, "rb") as f:
            linear_model = pickle.load(f)
        logistic_model = joblib.load(LOGISTIC_MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

    return linear_model, logistic_model, encoder, df, X_linear, y_linear

# ------------------------------
# Cálculo métricas
# ------------------------------
def calculate_metrics(linear_model, logistic_model, encoder, df, X_linear, y_linear):
    # Linear
    y_pred_linear = linear_model.predict(X_linear)
    rmse = np.sqrt(mean_squared_error(y_linear, y_pred_linear))

    # Logistic
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
        "y_true_linear": y_linear,
        "y_pred_linear": y_pred_linear
    }

# ------------------------------
# Execução do treino
# ------------------------------
csv_path = None
if uploaded_file:
    csv_path = os.path.join(project_root, uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.read())
    st.sidebar.success("✅ Arquivo CSV detectado e salvo.")

if retrain_btn:
    if not csv_path:
        train_status.error("❌ Nenhum arquivo CSV enviado.")
    else:
        linear_model, logistic_model, encoder, df, X_linear, y_linear = load_or_train_models(csv_path, retrain=True)
        metrics = calculate_metrics(linear_model, logistic_model, encoder, df, X_linear, y_linear)

        # Salvar no session_state
        st.session_state.update({
            "linear_model": linear_model,
            "logistic_model": logistic_model,
            "encoder": encoder,
            "metrics": metrics,
            "df": df,
            "X_linear": X_linear,
            "y_linear": y_linear
        })

# Recupera do session_state
linear_model = st.session_state.get("linear_model")
logistic_model = st.session_state.get("logistic_model")
encoder = st.session_state.get("encoder")
metrics = st.session_state.get("metrics")
df = st.session_state.get("df")
X_linear = st.session_state.get("X_linear")
y_linear = st.session_state.get("y_linear")

# ------------------------------
# Exibir métricas
# ------------------------------
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
        "y_true": metrics["y_true_linear"],
        "y_pred": metrics["y_pred_linear"]
    })
    fig_scatter = px.scatter(comp_df, x="y_true", y="y_pred",
                             title="Vendas Reais vs Previstas", trendline="ols")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ------------------------------
    # Previsão interativa
    # ------------------------------
    st.markdown("---")
    st.header("🔮 Previsão Interativa")
    st.subheader("Previsão de Vendas")
    item_weight = st.number_input("Peso do Item", min_value=0.0, step=0.1)
    item_visibility = st.number_input("Visibilidade do Item", min_value=0.0, step=0.01)
    item_mrp = st.number_input("Preço Máximo de Varejo", min_value=0.0, step=0.1)
    if st.button("Prever Vendas"):
        input_df = pd.DataFrame([[item_weight, item_visibility, item_mrp]],
                                columns=['Item_Weight','Item_Visibility','Item_MRP'])
        pred = linear_model.predict(input_df)[0]
        st.success(f"💰 Previsão de vendas: ${pred:.2f}")

    st.subheader("Previsão de Visibilidade")
    outlet_types = sorted(df['Outlet_Type'].dropna().unique())
    outlet_sizes = sorted(df['Outlet_Size'].dropna().unique())
    location_types = sorted(df['Outlet_Location_Type'].dropna().unique())
    outlet_type = st.selectbox("Tipo de Loja", outlet_types)
    outlet_size = st.selectbox("Tamanho da Loja", outlet_sizes)
    location_type = st.selectbox("Tipo de Localização", location_types)
    if st.button("Prever Visibilidade"):
        input_df = pd.DataFrame([[outlet_type, outlet_size, location_type]],
                                columns=['Outlet_Type','Outlet_Size','Outlet_Location_Type'])
        input_enc = encoder.transform(input_df)
        pred_vis = logistic_model.predict(input_enc)[0]
        st.success("✅ Visibilidade Alta" if pred_vis==1 else "⚠️ Visibilidade Baixa")

else:
    st.info("Carregue um arquivo CSV e clique em **Treinar modelo** para começar.")

# ------------------------------
# Chat Inteligente
# ------------------------------
st.markdown("---")
st.header("💬 Chat Inteligente")

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
