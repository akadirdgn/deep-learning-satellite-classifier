import tensorflow as tf
import os
from configs.settings import EPOCHS
from data.loader import load_data
from models.classifier import create_model
from data.database import init_db, DBLogCallback, save_prediction_log
from utils.visualization import plot_history, plot_confusion_matrix, plot_sample_images
from sklearn.metrics import classification_report
import numpy as np

import traceback

def train():
    try:
        # 1. Hazirlik
        print(">> Egitim hazirliklari basliyor...")
        init_db()
        
        # 2. Veri Yukleme
        train_ds, valid_ds, test_ds, class_names = load_data()
        
        # 3. Model Olusturma
        model = create_model(fine_tune=False) # Istege bagli True yapilabilir
        model.summary()
        
        # 4. Callbacks (Geri Cagrilar)
        # Egitim sirasinda devreye girecek yardimci fonksiyonlar
        callbacks = [
            # En iyi modeli kaydet (.keras formatinda)
            tf.keras.callbacks.ModelCheckpoint(
                filepath='model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            # Egitim kotuye giderse erken durdur (Sabir: 10 epoch)
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Ogrenme hizi (Learning Rate) takilip kalirsa azalt
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            # Veritabanina logla
            DBLogCallback()
        ]
        
        # 5. EGITIM (TRAINING)
        print(f"\n>> Egitim basliyor! Hedef: {EPOCHS} Epoch.")
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2
        )
        
        # 6. Sonuclar
        print("\n>> Egitim tamamlandi.")
        
        # En iyi model ile test seti performansi
        print(">> Test seti uzerinde degerlendiriliyor...")
        loss, acc = model.evaluate(test_ds)
        print(f">> Final Test Basarisi: %{acc*100:.2f}")
        
        # Her ihtimale karsi final modelini kaydet
        model.save('model.keras')
        print(">> Model 'model.keras' olarak kaydedildi.")
        
        # Grafikleri ciz
        plot_history(history)
        
        # ---------------------------------------------------------
        # EK: MODEL SUREKLI HATA VERDIGI ICIN DEGERLENDIRME BURADA YAPILIYOR
        # ---------------------------------------------------------
        print("\n>> Egitim sonrasi detayli degerlendirme basliyor (Confusion Matrix)...")
        
        # Tahminleri Al
        y_pred = []
        y_true = []
        
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1)) 
            y_true.extend(labels.numpy())
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Rapor
        print("\n" + "="*60)
        print("SINIFLANDIRMA RAPORU")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Ornek Resimler (Eğer evaluate calismazsa diye buraya da ekleyelim)
        plot_sample_images(dataset=None, class_names=class_names)
        
        # Veritabanina ornek kayit
        for i in range(10):
            real_name = class_names[y_true[-i]]
            pred_name = class_names[y_pred[-i]]
            save_prediction_log(real_name, pred_name)
            
        print(">> Tum ciktilar 'outputs/' klasorune kaydedildi.")

    except Exception:
        print(">> EGITIM SIRASINDA HATA OLUSTU:")
        traceback.print_exc()

if __name__ == "__main__":
    train()
