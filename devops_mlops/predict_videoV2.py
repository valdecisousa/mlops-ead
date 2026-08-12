import os
import cv2
from ultralytics import YOLO

def processar_video_local(video_input_path, video_output_path):
    # 1. Carrega o modelo treinado
    model = YOLO("best_75.pt")
    
    # DIAGNÓSTICO DE CLASSES
    print("\n🔍 Classes carregadas do best.pt:")
    for class_id, class_name in model.names.items():
        print(f"   - ID {class_id}: {class_name}")
    print("-" * 40)

    # 2. Abre o vídeo original
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"❌ Erro crítico: Não foi possível abrir o vídeo '{video_input_path}'")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Se o FPS ou dimensões vierem zerados, definimos um padrão seguro
    if fps == 0: fps = 30
    
    print(f"📹 Propriedades do vídeo original: {width}x{height} a {fps} FPS")
    
    # 3. Codec Altamente Compatível: XVID para gerar um arquivo .avi
    # O Windows costuma falhar silenciosamente com 'mp4v', mas o 'XVID' sempre funciona
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("❌ Erro do Windows: O OpenCV não conseguiu inicializar o gravador de vídeo.")
        return

    print(f"🎬 Processando frames de: {video_input_path}...")
    
    # Executa a predição frame a frame
    results = model.predict(
        source=video_input_path,
        save=False,          
        conf=0.70,           # Confiança baixa para testar se aparecem outras classes
        device="cpu",
        stream=True,         
        vid_stride=1,
        iou=0.5
    )
    
    contagem_frames = 0
    # 4. Loop de gravação forçada
    for r in results:
        # Força o desenho das caixas (labels) no frame
        frame_anotado = r.plot()
        
        # Garante que o frame anotado tenha o tamanho exato esperado pelo gravador
        frame_redimensionado = cv2.resize(frame_anotado, (width, height))
        
        # Grava fisicamente o frame
        out.write(frame_redimensionado)
        contagem_frames += 1
    
    # Fecha tudo e força a escrita no HD
    cap.release()
    out.release()
    
    print(f"📊 Total de frames processados pela IA: {contagem_frames}")
    
    # 5. Validação Real do Arquivo
    if os.path.exists(video_output_path) and os.path.getsize(video_output_path) > 0:
        print(f"\n✨ SUCESSO! Vídeo gerado com segurança em: '{video_output_path}'")
        print(f"Tamanho do arquivo: {os.path.getsize(video_output_path) / (1024*1024):.2f} MB")
    else:
        print(f"\n❌ Falha grave: O script rodou, mas o arquivo '{video_output_path}' foi bloqueado pelo sistema ou gerou vazio.")

if __name__ == "__main__":
    VIDEO_ENTRADA = "fab.mp4" 
    
    # Mudamos a extensão para .avi para garantir compatibilidade total com o OpenCV local
    VIDEO_SAIDA = "video_processado20.avi"
    
    if not os.path.exists(VIDEO_ENTRADA):
        print(f"❌ Erro: Arquivo de entrada '{VIDEO_ENTRADA}' não está na raiz do projeto.")
    else:
        processar_video_local(VIDEO_ENTRADA, VIDEO_SAIDA)
