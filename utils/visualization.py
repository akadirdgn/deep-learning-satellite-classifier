import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ==========================================
# GRAFIK VE GORSELLESTIRME
# ==========================================

def plot_history(history):
    """Egitim ve Dogrulama (Train/Val) grafiklerini cizer."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    
    # Basari Grafigi
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Egitim Basarisi')
    plt.plot(epochs_range, val_acc, label='Dogrulama Basarisi')
    plt.title('Basari Orani (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(loc='lower right')

    # Kayip Grafigi
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Egitim Kaybi')
    plt.plot(epochs_range, val_loss, label='Dogrulama Kaybi')
    plt.title('Kayip (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Grafigi kaydet
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    save_path = os.path.join('outputs', 'training_history.png')
    plt.savefig(save_path)
    print(f">> Grafik kaydedildi: {save_path}")
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Karmasiklik Matrisini (Confusion Matrix) cizdirir."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gercek Deger')
    plt.title('Karmasiklik Matrisi (Confusion Matrix)')
    
    # Grafigi kaydet
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    save_path = os.path.join('outputs', 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f">> Matris kaydedildi: {save_path}")
    # plt.show()

import random
import matplotlib.image as mpimg
from configs.settings import DATA_DIR, IMG_SIZE

def plot_sample_images(dataset=None, class_names=None):
    """
    Veri setinden her siniftan örnekleri (Train ve Valid) gorsellestirir.
    Dataset objesi yerine dogrudan dosya sisteminden okuyarak cesitliligi garanti eder.
    """
    if class_names is None:
        return

    num_classes = len(class_names)
    # 10 sinif var. Her biri icin 1 Train + 1 Valid gosterelim.
    # Toplam 20 resim. 4 satir x 5 sutun = 20 kutu.
    
    plt.figure(figsize=(15, 12))
    
    # Grid duzeni: 
    # Ilk 10 kare: Egitim (Train) verisinden her siniftan 1 tane
    # Sonraki 10 kare: Dogrulama (Valid) verisinden her siniftan 1 tane
    
    splits = ['train', 'valid']
    
    # 20'lik dongu kurmak yerine, once Train setini (10 sinif), sonra Valid setini (10 sinif) donelim
    plot_idx = 1
    
    for split in splits:
        for class_name in class_names:
            # Klasor yolu
            folder_path = os.path.join(DATA_DIR, split, class_name)
            
            if os.path.exists(folder_path):
                # Klasordeki tum resimler
                images = os.listdir(folder_path)
                if images:
                    # Rastgele bir resim sec
                    random_img_name = random.choice(images)
                    img_path = os.path.join(folder_path, random_img_name)
                    
                    # Resmi oku
                    img = mpimg.imread(img_path)
                    
                    # Cizdir
                    if plot_idx <= 20: # Guvenlik onlemi (max 20)
                        ax = plt.subplot(4, 5, plot_idx)
                        plt.imshow(img)
                        # Sadece sinif ismi, train/valid bilgisi kaldirildi
                        plt.title(f"{class_name}", fontsize=9)
                        plt.axis("off")
                        plot_idx += 1
            else:
                print(f"UYARI: Klasor bulunamadi: {folder_path}")
            
    plt.suptitle(f"Veri Setinden Ornekler (Sample Images)", fontsize=16)
    plt.tight_layout()
    
    # Grafigi kaydet
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        
    save_path = os.path.join('outputs', 'sample_images.png')
    plt.savefig(save_path)
    print(f">> Ornek resimler kaydedildi: {save_path}")

