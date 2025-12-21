import tensorflow as tf
import numpy as np
import os
from data.loader import load_data
from utils.visualization import plot_confusion_matrix, plot_sample_images
from data.database import save_prediction_log
from sklearn.metrics import classification_report
from models.classifier import create_model # Needed if we wanted to build, but with load_model we might not need it if not using weights only. 
# Better: Just use standard load_model
import tensorflow as tf

def evaluate():
    print(">> Degerlendirme Modu Baslatiliyor...")

    # 1. Veriyi Yukle (MODEL YUKLEMEDEN ONCE)
    # Sadece test verisine ihtiyacimiz var ama load_data hepsini donduruyor
    _, _, test_ds, class_names = load_data()

    # YENI: Ornek resimleri ciz ve kaydet (Hemen baslangicta)
    print(">> DEBUG: plot_sample_images cagiriliyor...")
    try:
        plot_sample_images(dataset=None, class_names=class_names)
        print(">> DEBUG: plot_sample_images tamamlandi.")
    except Exception as e:
        print(f">> DEBUG: plot_sample_images HATASI: {e}")
    
    model_path = 'model.keras'
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' dosyasi bulunamadi. Once egitimi calistirin (train.py).")
        return

    # 1. Modeli Yukle
    print(f">> Model yukleniyor: {model_path}")
    try:
        # Once normal yuklemeyi dene
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f">> UYARI: Normal model yukleme basarisiz ({e}). Agirlik yukleme (fallback) deneniyor...")
        # Basarisiz olursa mimariyi olusturup agirliklari yukle
        model = create_model(fine_tune=False)
        model.load_weights(model_path) # .keras dosyasindan agirliklari alir
    
    # 3. Tahminleri Al
    
    # 3. Tahminleri Al
    print(">> Tahminler yapiliyor...")
    y_pred = [] # Tahmin edilenler
    y_true = [] # Gercekler
    
    # Test veri setini donguye al
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1)) # Olasilik vektorunden sinifi sec
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 4. Raporlama
    print("\n" + "="*60)
    print("SINIFLANDIRMA RAPORU")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 5. Veritabanina Ornek Loglama (Son 10 tanesi)
    print(">> Veritabanina ornek tahminler kaydediliyor...")
    for i in range(10):
        # Rastgele 10 tanesini secmek yerine sondan 10 tanesini alalim
        real_name = class_names[y_true[-i]]
        pred_name = class_names[y_pred[-i]]
        save_prediction_log(real_name, pred_name)
        
    # 6. Grafik
    plot_confusion_matrix(y_true, y_pred, class_names)
    print("\n>> Degerlendirme tamamlandi.")

if __name__ == "__main__":
    evaluate()
