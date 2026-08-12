import os
from ultralytics import YOLO

def processar_video_local(video_input_path):
    # 1. Carrega o peso correto baixado do Kaggle (sempre 'best.pt')
    model = YOLO("best_75.pt")
    
    print(f"🎬 Iniciando a detecção por rede neural no vídeo: {video_input_path}")
    print("⚠️ Processando via CPU. Isso pode levar alguns minutos dependendo do tamanho do vídeo...")
    
    # Executa a predição com gerador (stream=True) para economizar memória RAM na CPU
    results = model.predict(
        source=video_input_path,
        save=True,           # Desenha as caixas delimitadoras e nomes na tela automaticamente
        conf=0.45,           # Subi para 30% para filtrar os erros da classe 'Bird' (pássaros)
        device="cpu",        # Força o uso da CPU local de forma estável
        stream=True,         # 🔥 Ativado: Processa frame por frame sem estourar a RAM do PC
        vid_stride=1         # Processa cada frame individualmente
    )
    
    # Como stream=True usa um gerador, precisamos consumir o gerador para o YOLO processar o vídeo
    for r in results:
        pass  # O YOLO já salva os frames automaticamente no disco por causa do save=True
    
    print("\n✨ Processamento concluído com sucesso!")
    print("📂 Abra a pasta 'runs/detect/predict/' para assistir ao vídeo anotado pela IA.")

if __name__ == "__main__":
    # Coloque o nome do vídeo que você jogou na pasta do projeto
    VIDEO_ENTRADA = "fab.mp4" 
    
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"❌ Erro: Coloque um arquivo de vídeo com o nome '{VIDEO_ENTRADA}' na raiz do seu projeto antes de rodar.")
    else:
        # Removido o segundo parâmetro porque o YOLO decide o nome e destino do output sozinho
        processar_video_local(VIDEO_ENTRADA)
