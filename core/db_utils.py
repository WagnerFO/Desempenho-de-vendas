import mysql.connector
import os

# 1. Criar conexão
def get_connection(database=None):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="sua_senha_aqui",
        database=database
    )

# 2. Criar database
def create_database(db_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    conn.commit()
    cursor.close()
    conn.close()

# 3. Dropar database
def drop_database(db_name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
    conn.commit()
    cursor.close()
    conn.close()

# 4. Executar arquivos SQL
def execute_sql_file(db_name, file_path):
    conn = get_connection(db_name)
    cursor = conn.cursor()
    with open(file_path, "r", encoding="utf-8") as f:
        sql_commands = f.read()
    for stmt in sql_commands.split(";"):
        if stmt.strip():
            cursor.execute(stmt)
    conn.commit()
    cursor.close()
    conn.close()
