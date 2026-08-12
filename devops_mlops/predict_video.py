import os
import cv2
from ultralytics import YOLO

def processar_video_local(video_input_path, video_output_path):
    # 1. Carrega o peso inteligente treinado no Kaggle
    model = YOLO("best_75.pt")
    
    # Como o Windows está usando CPU, o YOLO vai processar frame por frame de forma linear.
    # 'save=True' instrui a rede neural a desenhar as caixas e salvar o vídeo automaticamente.
    print(f"🎬 Iniciando a detecção por rede neural no vídeo: {video_input_path}")
    results = model.predict(
        source=video_input_path,
        save=True,           # Desenha as caixas delimitadoras e nomes na tela automaticamente
        conf=0.30,           # Filtro de confiança mínima de 25% para evitar falsos positivos
        device="cpu",         # Força o uso da CPU local de forma estável
        #stream=True,         # 🔥 CHAVE DE OURO: Ativa o modo fluxo contínuo contra estouro de RAM
        vid_stride=1         # Processa cada frame individualmente
    )
    
    # O YOLO salva por padrão na pasta incremental 'runs/detect/predict/'
    # Vamos apenas avisar onde o arquivo final foi parar
    print("\n✨ Processamento concluído com sucesso!")
    print("Abra a pasta 'runs/detect/predict/' no seu VS Code para assistir ao vídeo anotado pela IA.")

if __name__ == "__main__":
    # Coloque o nome do vídeo que você jogou na pasta do projeto
    VIDEO_ENTRADA = "fab.mp4" 
    
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"❌ Erro: Coloque um arquivo de vídeo com o nome '{VIDEO_ENTRADA}' na raiz do seu projeto antes de rodar.")
    else:
        processar_video_local(VIDEO_ENTRADA, "video_processado75.mp4")
