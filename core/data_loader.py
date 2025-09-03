import pandas as pd
from core.db_utils import get_connection

# 5. Função para inserir dados de CSV em tabela
def insert_csv_to_table(db_name, csv_path, table_name):
    conn = get_connection(db_name)
    cursor = conn.cursor()

    df = pd.read_csv(csv_path)

    # Preparar query INSERT dinâmica
    cols = ", ".join(df.columns)
    placeholders = ", ".join(["%s"] * len(df.columns))
    insert_stmt = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders})"

    # Inserir linha a linha
    for row in df.itertuples(index=False, name=None):
        cursor.execute(insert_stmt, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()
