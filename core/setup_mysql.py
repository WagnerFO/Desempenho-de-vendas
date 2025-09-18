import os
import glob
import pandas as pd
from sqlalchemy import create_engine, text

# ========================
# CONFIGURAÇÃO DO BANCO
# ========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "4019")
DB_NAME = os.getenv("DB_NAME", "BigMarkSales")

# Pasta contendo os arquivos .sql
SQL_SCRIPTS_DIR = os.getenv(
    "SQL_SCRIPTS_DIR",
    r"C:\Users\wagne\IntelliJ Idea-Workspace\Desempenho-de-vendas\core\data"
)

# ========================
# FUNÇÕES
# ========================
def get_engine():
    """Cria e retorna o engine SQLAlchemy para o banco"""
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    return create_engine(url, future=True)


def create_database_if_not_exists():
    """Cria o banco se não existir"""
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/"
    engine = create_engine(url, future=True)
    with engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    engine.dispose()


def execute_sql_folder(sql_dir: str = SQL_SCRIPTS_DIR):
    """Executa todos os .sql da pasta"""
    engine = get_engine()
    sql_files = sorted(glob.glob(os.path.join(sql_dir, "*.sql")))
    if not sql_files:
        print(f"[AVISO] Nenhum .sql encontrado em {sql_dir}")
        return

    with engine.begin() as conn:
        for file in sql_files:
            print(f"[EXEC] {file}")
            with open(file, "r", encoding="utf-8") as f:
                sql_text = f.read()
                conn.execute(text(sql_text))
    engine.dispose()


def insert_dataframe(df: pd.DataFrame, table_name: str, if_exists="replace"):
    """Insere DataFrame no banco"""
    engine = get_engine()
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    engine.dispose()
