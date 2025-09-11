import os
import pandas as pd

# Diretório para salvar os dados tratados
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

def save_dataframe(df: pd.DataFrame, name: str):
    """Salva um DataFrame tratado em CSV dentro de data/processed"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"✅ DataFrame salvo em {path}")

def load_dataframe(name: str) -> pd.DataFrame:
    """Carrega um DataFrame tratado previamente salvo"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo {path} não encontrado!")
    return pd.read_csv(path)
