import os
import mlflow
from ultralytics import YOLO

def train_detection_model():
    # 1. Configurar o MLflow para usar sua pasta 'mlruns' existente
    # Caminho absoluto ou relativo apontando para a raiz onde está sua pasta mlruns
    mlruns_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
    mlflow.set_tracking_uri(f"file://{mlruns_path}")
    mlflow.set_experiment("Detecao_Aeronaves_YOLO")

    # 2. Inicializar o modelo YOLO pré-treinado (Transfer Learning)
    # Usamos o 'yolov8n.pt' (nano) pois ele treina muito rápido, ideal para computadores comuns
    # e evita overfitting em datasets menores.
    model = YOLO("yolov8n.pt")

    # 3. Iniciar a run do MLflow para envelopar o treinamento
    with mlflow.start_run(run_name="yolov8n_com_albumentations") as run:
        
        print("Iniciando treinamento com rastreamento MLOps ativo...")
        
        # 4. Executar o treinamento com parâmetros anti-overfitting
        results = model.train(
            data="configs/dataset.yaml", # Arquivo de configuração das classes
            epochs=50,                  # Máximo de épocas
            patience=10,                 # EARLY STOPPING: Se em 10 épocas o mAP não melhorar, para!
            batch=16,                    # Tamanho do lote de imagens
            imgsz=640,                   # Resolução padrão de redimensionamento
            workers=2,                   # Threads para carregar dados
            device="cpu",                # Mude para "cuda" ou 0 se tiver GPU Nvidia dedicada
            weight_decay=0.0005,         # Regularização L2 para penalizar pesos extremos (anti-overfitting)
            label_smoothing=0.1,         # Evita que o modelo fique "confiante demais" nas marcações
            plots=True                   # Gera matriz de confusão e curvas F1/Precision automaticamente
        )
        
        # O Ultralytics salva automaticamente em 'runs/detect/train/' os pesos (.pt)
        # Vamos registrar explicitamente o melhor modelo gerado dentro do MLflow Artifacts
        best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model_weights")
            print(f"Sucesso! Melhor modelo salvo no MLflow e localmente em: {best_model_path}")

if __name__ == "__main__":
    train_detection_model()
