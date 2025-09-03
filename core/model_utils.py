import pickle
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Função para treinar e salvar um modelo
def train_and_save_model(features: pd.DataFrame, labels: pd.Series, model_type="linear", model_name="model"):
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Tipo de modelo não suportado. Use 'linear' ou 'logistic'.")

    # Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Treinar modelo
    model.fit(X_train, y_train)

    # Avaliar rapidamente
    if model_type == "linear":
        preds = model.predict(X_test)
        score = mean_squared_error(y_test, preds, squared=False)  # RMSE
        print(f"[{model_type}] RMSE: {score:.2f}")
    else:
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(f"[{model_type}] Accuracy: {score:.2%}")

    # Salvar modelo na pasta /model
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", f"{model_name}.pickle")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo salvo em {model_path}")
