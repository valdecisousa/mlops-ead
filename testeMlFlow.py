import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carrega dataset
data = load_iris()

X, y = data.data, data.target

# Divide dados
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Inicia execução
with mlflow.start_run():

    # Modelo
    model = LogisticRegression(max_iter=1000)

    # Treinamento
    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)

    # Métrica
    accuracy = accuracy_score(y_test, y_pred)

    # Logs
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)

    # Salva modelo
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )

    print("Acurácia:", accuracy)