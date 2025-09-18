# setup_mysql.py
# ------------------------------------------------------------
# O QUE ESTE SCRIPT FAZ
# 1) Cria um banco de dados MySQL (local) se não existir.
# 2) Executa todos os arquivos .sql (CREATE TABLE etc.) de uma pasta.
# 3) Insere DataFrames do pandas em tabelas do banco (função pronta).
#
# PRÉ-REQUISITOS
# - MySQL Server instalado e rodando (Workbench é apenas a GUI).
# - Python 3.9+ com os pacotes:
#     pip install mysql-connector-python SQLAlchemy pandas
#
# COMO USAR (exemplo rápido)
#   python setup_mysql.py
#
# Você pode adaptar as variáveis de conexão e o caminho da pasta de .sql
# diretamente abaixo em CONFIG.
# ------------------------------------------------------------

from __future__ import annotations
import os
import glob
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine, text
import pandas as pd
from typing import Optional, Dict, Any

# =========================
# CONFIG
# =========================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "root")        # ajuste aqui
DB_NAME = os.getenv("DB_NAME", "meu_banco")   # nome do banco a criar

# Pasta contendo os arquivos .sql com os CREATE TABLE/INDEX/etc.
# Ex.: "C:/Users/Voce/projeto/sql" ou "./sql_creates"
SQL_SCRIPTS_DIR = os.getenv("SQL_SCRIPTS_DIR", "./sql_creates")


# =========================
# FUNÇÕES PRINCIPAIS
# =========================
def create_database_if_not_exists(
    host: str, port: int, user: str, password: str, db_name: str
) -> None:
    """
    Conecta no servidor MySQL (sem selecionar DB) e cria o banco se não existir.
    Define utf8mb4 como charset padrão.
    """
    try:
        conn = mysql.connector.connect(
            host=host, port=port, user=user, password=password
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci"
        )
        cur.close()
        conn.close()
        print(f"[OK] Banco '{db_name}' garantido (criado se não existia).")
    except mysql.connector.Error as err:
        raise RuntimeError(f"Erro ao criar banco: {err}") from err


def get_sqlalchemy_engine(
    host: str, port: int, user: str, password: str, db_name: str
):
    """
    Cria um engine SQLAlchemy usando o driver mysql-connector-python.
    Esse engine é usado pelo pandas.to_sql para inserir DataFrames.
    """
    url = (
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
        "?charset=utf8mb4"
    )
    engine = create_engine(url, future=True)
    return engine


def execute_sql_folder(
    host: str,
    port: int,
    user: str,
    password: str,
    db_name: str,
    sql_dir: str,
) -> None:
    """
    Executa todos os .sql da pasta `sql_dir` dentro do banco `db_name`, na ordem alfabética.
    Usa o mysql-connector com multi=True, então o arquivo pode conter vários statements.
    """
    if not os.path.isdir(sql_dir):
        raise FileNotFoundError(
            f"Pasta '{sql_dir}' não encontrada. Ajuste SQL_SCRIPTS_DIR."
        )

    sql_files = sorted(glob.glob(os.path.join(sql_dir, "*.sql")))
    if not sql_files:
        print(f"[AVISO] Nenhum .sql encontrado em {sql_dir}.")
        return

    try:
        conn = mysql.connector.connect(
            host=host, port=port, user=user, password=password, database=db_name
        )
        cur = conn.cursor()
        for path in sql_files:
            with open(path, "r", encoding="utf-8") as f:
                sql_text = f.read()
            print(f"[EXEC] {os.path.basename(path)}")
            # multi=True permite múltiplos comandos no mesmo arquivo
            for _ in cur.execute(sql_text, multi=True):
                pass
        conn.commit()
        cur.close()
        conn.close()
        print("[OK] Todos os arquivos .sql foram executados com sucesso.")
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_DBACCESS_DENIED_ERROR:
            msg = "Acesso negado ao banco. Verifique usuário/senha/permissões."
        else:
            msg = str(err)
        raise RuntimeError(f"Erro ao executar .sql: {msg}") from err


def insert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    engine,
    if_exists: str = "append",
    chunksize: int = 1000,
    dtype: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Insere um DataFrame em `table_name` usando pandas.to_sql.
    - if_exists: 'append' (padrão), 'replace', 'fail'
    - chunksize: controla o tamanho dos lotes de inserção
    - dtype: mapeamento opcional de colunas para tipos SQL (ex: {"col": sqlalchemy.types.VARCHAR(255)})

    Retorna o número de linhas inseridas.
    """
    if df.empty:
        print(f"[SKIP] DataFrame vazio para '{table_name}'. Nada a inserir.")
        return 0

    # Garante colunas como str (nome da tabela não aceita aspas em to_sql com MySQL)
    table = table_name.strip("`")

    with engine.begin() as conn:
        # Para MySQL, usar method='multi' melhora performance.
        df.to_sql(
            name=table,
            con=conn,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
            dtype=dtype,
        )
    print(f"[OK] Inseridas {len(df)} linhas em '{table}'.")
    return len(df)


# =========================
# EXEMPLO DE USO
# =========================
def main():
    # 1) Cria o banco se não existir
    create_database_if_not_exists(DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME)

    # 2) Executa os CREATE TABLE (e outros) a partir dos .sql na pasta
    execute_sql_folder(DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME, SQL_SCRIPTS_DIR)

    # 3) Insere DataFrames de exemplo (substitua pelos seus)
    engine = get_sqlalchemy_engine(DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME)

    # EXEMPLO 1: DataFrame para tabela 'clientes'
    df_clientes = pd.DataFrame(
        [
            {"id": 1, "nome": "Ana", "email": "ana@example.com"},
            {"id": 2, "nome": "Bruno", "email": "bruno@example.com"},
        ]
    )
    insert_dataframe(df_clientes, table_name="clientes", engine=engine, if_exists="append")

    # EXEMPLO 2: DataFrame para tabela 'pedidos'
    df_pedidos = pd.DataFrame(
        [
            {"id": 100, "cliente_id": 1, "valor": 199.90},
            {"id": 101, "cliente_id": 2, "valor": 89.50},
        ]
    )
    insert_dataframe(df_pedidos, table_name="pedidos", engine=engine, if_exists="append")

    # (Opcional) Verificação simples com SELECT
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) AS n FROM clientes"))
        print("[CHECK] Total de clientes:", result.scalar())


if __name__ == "__main__":
    main()