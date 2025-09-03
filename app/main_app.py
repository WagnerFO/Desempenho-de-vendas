import os
import pandas as pd
from core import db_utils, data_loader, model_utils

DB_NAME = "ml_retail_db"

def main():
    # 1. Criar database
    print("Criando database...")
    db_utils.create_database(DB_NAME)

    # 2. Executar scripts SQL (SOR → SOT → SPEC)
    sql_folder = os.path.join("core", "data")
    sql_files = [
        "sor_items.sql", "sor_outlets.sql", "sor_sales.sql",
        "sot_items.sql", "sot_outlets.sql", "sot_sales.sql",
        "spec_features.sql", "spec_labels.sql"
    ]
    for file in sql_files:
        path = os.path.join(sql_folder, file)
        db_utils.execute_sql_file(DB_NAME, path)

    # 3. Inserir dados nas tabelas SOR
    data_loader.insert_csv_to_table(DB_NAME, "Train.csv", "sor_items")
    data_loader.insert_csv_to_table(DB_NAME, "Train.csv", "sor_outlets")
    data_loader.insert_csv_to_table(DB_NAME, "Train.csv", "sor_sales")

    # ⚠️ Aqui deveria entrar a lógica para transformar SOR → SOT → SPEC
    # Por enquanto, vamos simular SPEC com Pandas
    
    df = pd.read_csv("Train.csv")

    # --------------------------
    # Modelo Linear (previsão de vendas)
    # --------------------------
    features_linear = df[["Item_MRP", "Item_Weight"]].fillna(0)
    labels_linear = df["Item_Outlet_Sales"]

    model_utils.train_and_save_model(
        features_linear, labels_linear,
        model_type="linear",
        model_name="sales_regression"
    )

    # --------------------------
    # Modelo Logístico (classificação de visibilidade)
    # --------------------------
    # Criar alvo binário: 1 = visibilidade acima da mediana, 0 = abaixo
    median_visibility = df["Item_Visibility"].median()
    labels_logistic = (df["Item_Visibility"] > median_visibility).astype(int)

    # Features para o modelo logístico
    features_logistic = df[["Outlet_Type", "Outlet_Size", "Outlet_Location_Type"]].fillna("Unknown")

    # One-hot encoding para variáveis categóricas
    features_logistic = pd.get_dummies(features_logistic, drop_first=True)

    model_utils.train_and_save_model(
        features_logistic, labels_logistic,
        model_type="logistic",
        model_name="visibility_classifier"
    )

    # 5. Dropar DB (apenas se você quiser limpa
