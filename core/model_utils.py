import pickle
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

# Função para treinar o modelo de Regressão Linear
def train_linear_model(features: pd.DataFrame, labels: pd.Series):
    """
    Treina e salva um modelo de Regressão Linear para previsão de vendas.
    """
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = np.sqrt(mean_squared_error(y_test, preds)) # RMSE
    print(f"[Linear Regression] RMSE: {score:.2f}")
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "linear_regression_model.pickle")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

# Função para treinar o modelo de Regressão Logística
def train_logistic_model(features: pd.DataFrame, labels: pd.Series):
    """
    Treina e salva um modelo de Regressão Logística para classificação.
    Também salva o encoder de dados categóricos.
    """
    # Pré-processamento: One-Hot Encoding
    try:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore')
    
    X_encoded = encoder.fit_transform(features)
    
    # Treinamento do modelo
    model = LogisticRegression(max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliação do modelo
    preds = model.predict(X_test)
    score = accuracy_score(y_test, preds)
    print(f"[Logistic Regression] Accuracy: {score:.2%}")
    
    # Salvar o modelo e o encoder
    os.makedirs("model", exist_ok=True)
    model_path = os.path.join("model", "logistic_regression_model.pickle")
    joblib.dump(model, model_path)
    encoder_path = os.path.join("model", "onehot_encoder.joblib")
    joblib.dump(encoder, encoder_path)
    
    return model