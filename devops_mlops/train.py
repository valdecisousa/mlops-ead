import os
import mlflow
from ultralytics import YOLO

def train_model(config_path="configs/dataset.yaml", epochs=50):
    # Configuração de caminhos estável para o ambiente do Kaggle
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Deteccao_Aeronaves_YOLO")

    model = YOLO("yolov8n.pt")

    with mlflow.start_run(run_name="treino_completo_50_epocas") as run:
        print("🚀 Iniciando treinamento completo de 50 épocas na GPU...")
        
        results = model.train(
            data=config_path,
            epochs=epochs,              # 🔥 50 Épocas configuradas direto na raiz
            patience=12,                
            batch=32,                   
            imgsz=640,                  
            device='cuda',              # 🔥 Força o uso das duas GPUs T4 do Kaggle
            weight_decay=0.0005, 
            label_smoothing=0.1, 
            plots=True           
        )
        
        best_model_path = os.path.join(results.save_dir, "weights", "best_50.pt")
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model_weights")
            print(f"✨ Sucesso! Melhor modelo salvo no MLflow em: {best_model_path}")
            
    return results

if __name__ == "__main__":
    train_model()
