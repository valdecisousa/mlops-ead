import os
import random
import shutil

# --- CONFIGURAÇÃO DO USUÁRIO ---
PASTA_FONTE_IMGS = r"C:\Users\valt_\Downloads\dataset_aeronaves_v2\Images"
PASTA_FONTE_LABELS = r"C:\Users\valt_\Downloads\dataset_aeronaves_v2\Labels"

PASTA_DESTINO = "dataset_aeronaves_fatia"  # Nova pasta para zipar

# Quantidade de imagens que você quer extrair por etapa
AMOSTRAS_TREINO = 2000
AMOSTRAS_VALID = 400
AMOSTRAS_TESTE = 0  # Configurado como 0, pois o seu YAML joga a validação na pasta test

# Configuração dinâmica para as subpastas adicionais do seu dataset
SUBPASTAS_INTERNAS = {
    "train": {
        "qtd": AMOSTRAS_TREINO, 
        "imgs": "train", 
        "lbls": os.path.join("train", "labels")
    },
    "valid": {
        "qtd": AMOSTRAS_VALID, 
        "imgs": "valid", 
        "lbls": os.path.join("valid", "labels")
    },
    "test": {
        "qtd": AMOSTRAS_TESTE, 
        "imgs": "test", 
        "lbls": os.path.join("test", "labels")
    }
}

def criar_fatia_dataset():
    random.seed(42)  # Garante que o sorteio seja o mesmo se rodar de novo
    
    # Remove a pasta de destino se ela já existir de uma execução anterior
    if os.path.exists(PASTA_DESTINO):
        shutil.rmtree(PASTA_DESTINO)
        
    for etapa, config_etapa in SUBPASTAS_INTERNAS.items():
        qtd_alvo = config_etapa["qtd"]
        
        # Se a quantidade de amostras for 0, pula a etapa (como o 'test' configurado acima)
        if qtd_alvo <= 0:
            continue
            
        src_img_dir = os.path.join(PASTA_FONTE_IMGS, config_etapa["imgs"])
        src_lbl_dir = os.path.join(PASTA_FONTE_LABELS, config_etapa["lbls"])
        
        if not os.path.exists(src_img_dir):
            print(f"⚠️ Pasta de imagens {src_img_dir} não encontrada. Pulando...")
            continue
        if not os.path.exists(src_lbl_dir):
            print(f"⚠️ Pasta de rótulos {src_lbl_dir} não encontrada. Pulando...")
            continue
            
        # Lista todas as imagens disponíveis na pasta
        todas_imagens = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Garante que não vai pedir mais do que existe
        qtd_selecionar = min(qtd_alvo, len(todas_imagens))
        imagens_selecionadas = random.sample(todas_imagens, qtd_selecionar)
        
        # Define o nome da pasta de destino com base no mapeamento do seu dataset.yaml do Kaggle
        # Seu YAML aponta 'val' para 'test/images'. Portanto, a pasta 'valid' do AOD vai para 'test'
        pasta_mapeada = "train" if etapa == "train" else "test"
        
        dest_img_dir = os.path.join(PASTA_DESTINO, "data", "processed", pasta_mapeada, "images")
        dest_lbl_dir = os.path.join(PASTA_DESTINO, "data", "processed", pasta_mapeada, "labels")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_lbl_dir, exist_ok=True)
        
        print(f"🚚 Separando {qtd_selecionar} arquivos casados para a etapa de origem: {etapa} -> Destino: {pasta_mapeada}")
        
        copiados = 0
        for img_name in imagens_selecionadas:
            base_name, _ = os.path.splitext(img_name)
            lbl_name = f"{base_name}.txt"
            
            caminho_img_origem = os.path.join(src_img_dir, img_name)
            caminho_lbl_origem = os.path.join(src_lbl_dir, lbl_name)
            
            # Só copia se o arquivo .txt correspondente existir de verdade na subpasta /labels
            if os.path.exists(caminho_lbl_origem):
                shutil.copy(caminho_img_origem, os.path.join(dest_img_dir, img_name))
                shutil.copy(caminho_lbl_origem, os.path.join(dest_lbl_dir, lbl_name))
                copiados += 1
                
        print(f"✅ Sucesso: {copiados} pares de arquivos integrados com sucesso.")

    # Gera o arquivo ZIP final pronto para o Kaggle automaticamente
    print("\n📦 Compactando os arquivos processados...")
    shutil.make_archive("projeto_fatia_mlops", 'zip', PASTA_DESTINO)
    
    # Limpa a pasta temporária gerada para economizar espaço no computador
    shutil.rmtree(PASTA_DESTINO)
    print("🎯 FIM: Arquivo 'projeto_fatia_mlops.zip' gerado com sucesso na raiz!")

if __name__ == "__main__":
    criar_fatia_dataset()
