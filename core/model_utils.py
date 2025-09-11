from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import pickle
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_linear_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    with open(os.path.join(MODEL_DIR, "linear_regression_model.pickle"), "wb") as f:
        pickle.dump(model, f)
    print("✅ Linear Regression salvo em model/")
    return model

def train_logistic_model(X, y):
    import inspect
    # Verifica se a versão do sklearn tem o parâmetro 'sparse_output'
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")  # sklearn >= 1.2
    else:
        encoder = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")  # sklearn < 1.2

    X_encoded = encoder.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_encoded, y)

    # salvar modelo e encoder
    with open(os.path.join(MODEL_DIR, "logistic_regression_model.pickle"), "wb") as f:
        pickle.dump(model, f)
    joblib.dump(encoder, os.path.join(MODEL_DIR, "onehot_encoder.joblib"))

    print("✅ Logistic Regression e encoder salvos em model/")
    return model

