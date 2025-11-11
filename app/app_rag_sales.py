import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect
from openai import OpenAI
from dotenv import load_dotenv
import json

# -------------------------------------------------------
# Configurações iniciais
# -------------------------------------------------------
st.set_page_config(page_title="IA de Desempenho de Vendas", page_icon="📊", layout="wide")

# Carrega variáveis de ambiente (API KEY)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Chave OpenAI não encontrada! Adicione um arquivo `.env` com OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Caminho do banco
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "BigMarkSales.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

# -------------------------------------------------------
# Funções utilitárias
# -------------------------------------------------------
def get_table_names(engine):
    inspector = inspect(engine)
    return inspector.get_table_names()

def summarize_table(engine, table_name, limit=5):
    """Retorna uma amostra e estatísticas básicas de uma tabela"""
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", engine)
    stats = df.describe(include="all").to_dict()
    return {"name": table_name, "sample": df.to_dict(orient="records"), "stats": stats}

def build_context(engine):
    """Gera um resumo geral de todas as tabelas do banco"""
    tables = get_table_names(engine)
    summary = {}
    for table in tables:
        summary[table] = summarize_table(engine, table)
    return summary

def ask_gpt(question, context):
    """Envia pergunta ao GPT com contexto dos dados"""
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um analista de dados experiente. "
                "Resuma e interprete informações sobre desempenho de vendas. "
                "Responda de forma técnica, mas compreensível, citando métricas e comparações sempre que possível."
            ),
        },
        {
            "role": "user",
            "content": f"Aqui está o resumo dos dados:\n{json.dumps(context, indent=2)}"
        },
        {
            "role": "user",
            "content": f"Pergunta do usuário: {question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# -------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------
st.title("📊 Análise Inteligente de Desempenho de Vendas")
st.markdown("Use o GPT para obter insights automáticos sobre os dados de vendas do banco **BigMarkSales.db**.")

st.divider()

# Exibir tabelas detectadas
tables = get_table_names(engine)
st.sidebar.header("Tabelas disponíveis")
st.sidebar.write(tables)

selected_table = st.sidebar.selectbox("Visualizar tabela:", tables)

if selected_table:
    df_preview = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 10", engine)
    st.subheader(f"📋 Prévia da tabela: `{selected_table}`")
    st.dataframe(df_preview)

st.divider()
st.subheader("💬 Consultar IA")

question = st.text_area("Digite sua pergunta sobre os dados de vendas:", placeholder="Ex: Quais fatores mais influenciam as vendas nas lojas?")
if st.button("Enviar para IA") and question:
    with st.spinner("Consultando IA..."):
        context = build_context(engine)
        answer = ask_gpt(question, context)
        st.markdown("### 🧠 Resposta da IA")
        st.write(answer)

st.caption("Desenvolvido com por Wagner e Clerissona — IA aplicada à análise de desempenho de vendas.")
