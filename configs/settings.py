import os

# ==========================================
# PROJE AYARLARI (CONFIGURATION)
# ==========================================

# Veri Yolu (Train/Test/Valid klasorlerinin oldugu yer)
DATA_DIR = "Datasets_Split"

# Goruntu Ayarlari
IMG_SIZE = (64, 64)    # MobileNetV2 icin uygun boyut
INPUT_SHAPE = (64, 64, 3)
NUM_CLASSES = 10

# Egitim Ayarlari
BATCH_SIZE = 64        # Bir kerede islenecek resim sayisi
EPOCHS = 5           # Toplam egitim turu
LEARNING_RATE = 0.001  # Ogrenme hizi (Adam optimizer default)

# Veritabani
DB_NAME = "project.db"

# Reproducibility (Tekrarlanabilirlik)
SEED = 42
