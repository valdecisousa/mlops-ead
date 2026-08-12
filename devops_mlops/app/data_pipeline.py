import os
import random
import shutil
import cv2
import albumentations as A

# 1. Configuração dos caminhos
RAW_IMG_DIR = "data/raw/images"
RAW_LBL_DIR = "data/raw/labels"
OUTPUT_DIR = "data/processed"

# 2. Definição do pipeline de Augmentation (Apenas para o Treino)
# O Albumentations atualiza as caixas delimitadoras de forma automática
transform_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def split_and_augment(train_ratio=0.8, augment_multiplier=3):
    """
    Separa os dados originais rigidamente e aplica aumento apenas no treino.
    """
    images = [f for f in os.listdir(RAW_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.seed(42) # Garante reprodutibilidade do split
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    # Processa conjunto de Teste (Cópia pura, sem alterações)
    print("Processando conjunto de teste (inalterado)...")
    for img_name in test_images:
        base_name = os.path.splitext(img_name)[0]
        # Copia imagem
        shutil.copy(os.path.join(RAW_IMG_DIR, img_name), os.path.join(OUTPUT_DIR, "test/images", img_name))
        # Copia rótulo correspondente
        lbl_name = f"{base_name}.txt"
        if os.path.exists(os.path.join(RAW_LBL_DIR, lbl_name)):
            shutil.copy(os.path.join(RAW_LBL_DIR, lbl_name), os.path.join(OUTPUT_DIR, "test/labels", lbl_name))

    # Processa conjunto de Treino com Data Augmentation
    print("Processando e aumentando conjunto de treino...")
    for img_name in train_images:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(RAW_IMG_DIR, img_name)
        lbl_path = os.path.join(RAW_LBL_DIR, f"{base_name}.txt")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ler bboxes do arquivo YOLO
        bboxes = []
        class_labels = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])
                    
        # Salva a imagem de treino original na pasta final
        shutil.copy(img_path, os.path.join(OUTPUT_DIR, "train/images", img_name))
        shutil.copy(lbl_path, os.path.join(OUTPUT_DIR, "train/labels", f"{base_name}.txt"))
        
        # Gera as versões aumentadas (Small Data Solution)
        for i in range(augment_multiplier):
            try:
                transformed = transform_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                
                # Salva nova imagem aumentada
                new_img_name = f"{base_name}_aug_{i}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "train/images", new_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                # Salva novo rótulo com bboxes recalculadas
                new_lbl_name = f"{base_name}_aug_{i}.txt"
                with open(os.path.join(OUTPUT_DIR, "train/labels", new_lbl_name), 'w') as f:
                    for cls, box in zip(class_labels, aug_bboxes):
                        f.write(f"{cls} {' '.join([str(coord) for coord in box])}\n")
            except Exception as e:
                # O Albumentations descarta bboxes que saem da tela na rotação extrema
                continue

if __name__ == "__main__":
    # Criar estrutura de diretórios de saída
    for folder in ["train/images", "train/labels", "test/images", "test/labels"]:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
        
    split_and_augment()
    print("Pipeline de dados concluído com sucesso!")
