# core/db_utils.py
import os
from sqlalchemy import create_engine, text, inspect

# --------------------------
# Configurações do banco
# --------------------------
DB_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
os.makedirs(DB_FOLDER, exist_ok=True)

def get_db_path(db_name: str) -> str:
    """Retorna o caminho completo do arquivo SQLite"""
    return os.path.join(DB_FOLDER, f"{db_name}.db")

def get_engine(db_name: str):
    """Retorna um SQLAlchemy Engine para o banco SQLite"""
    db_path = get_db_path(db_name)
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    return engine

# --------------------------
# 1. Criar database
# --------------------------
def create_database(db_name: str):
    """Cria o arquivo de banco SQLite se não existir"""
    db_path = get_db_path(db_name)
    if not os.path.exists(db_path):
        open(db_path, "w").close()  # cria arquivo vazio

# --------------------------
# 2. Dropar database
# --------------------------
def drop_database(db_name: str):
    """Deleta o arquivo do banco SQLite"""
    db_path = get_db_path(db_name)
    if os.path.exists(db_path):
        os.remove(db_path)

# --------------------------
# 3. Verificar se tabela existe
# --------------------------
def table_exists(engine, table_name: str) -> bool:
    """Verifica se uma tabela existe no banco"""
    inspector = inspect(engine)
    return inspector.has_table(table_name)

# --------------------------
# 4. Executar arquivos SQL
# --------------------------
def execute_sql_file(engine, file_path: str):
    """
    Executa um arquivo .sql inteiro no banco SQLite através do SQLAlchemy.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        sql_commands = f.read()

    commands = [cmd.strip() for cmd in sql_commands.split(";") if cmd.strip()]

    with engine.connect() as conn:
        for cmd in commands:
            conn.execute(text(cmd))
        conn.commit()
