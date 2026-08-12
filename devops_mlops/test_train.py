import os
import pytest

# Caminhos gerados pelo app/data_pipeline.py
PROCESSED_DIR = "data/processed"
TRAIN_IMG_DIR = os.path.join(PROCESSED_DIR, "train/images")
TRAIN_LBL_DIR = os.path.join(PROCESSED_DIR, "train/labels")
TEST_IMG_DIR = os.path.join(PROCESSED_DIR, "test/images")
TEST_LBL_DIR = os.path.join(PROCESSED_DIR, "test/labels")

def test_processed_directories_exist():
    """Garante que as pastas pós-processamento foram criadas."""
    assert os.path.exists(TRAIN_IMG_DIR), "Pasta de imagens de treino não encontrada."
    assert os.path.exists(TEST_IMG_DIR), "Pasta de imagens de teste não encontrada."
    assert os.path.exists(TRAIN_LBL_DIR), "Pasta de labels de treino não encontrada."
    assert os.path.exists(TEST_LBL_DIR), "Pasta de labels de teste não encontrada."

def test_data_augmentation_only_in_train():
    """Valida que o Data Augmentation NÃO afetou o teste (Exigência do enunciado)."""
    train_imgs = os.listdir(TRAIN_IMG_DIR)
    test_imgs = os.listdir(TEST_IMG_DIR)
    
    # Arquivos modificados pelo Albumentations contêm '_aug_' no nome
    aug_in_test = [img for img in test_imgs if "_aug_" in img]
    aug_in_train = [img for img in train_imgs if "_aug_" in img]
    
    assert len(aug_in_test) == 0, "❌ ERRO MLOps: Data Augmentation foi aplicado no conjunto de teste!"
    assert len(aug_in_train) > 0, "❌ ERRO: O pipeline de treino não gerou imagens aumentadas."

def test_image_label_match():
    """Garante a integridade do dataset: cada imagem precisa de um arquivo .txt de caixas."""
    for folder_img, folder_lbl in [(TRAIN_IMG_DIR, TRAIN_LBL_DIR), (TEST_IMG_DIR, TEST_LBL_DIR)]:
        images = {os.path.splitext(f)[0] for f in os.listdir(folder_img) if f.endswith(('.jpg', '.jpeg', '.png'))}
        labels = {os.path.splitext(f)[0] for f in os.listdir(folder_lbl) if f.endswith('.txt')}
        
        assert images == labels, f"❌ Descompasso em {folder_img}: imagens e rótulos não batem!"
