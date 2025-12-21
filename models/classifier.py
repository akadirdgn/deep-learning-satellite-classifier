import tensorflow as tf
from configs.settings import IMG_SIZE, NUM_CLASSES, INPUT_SHAPE, LEARNING_RATE

def create_model(fine_tune=False):
    """
    MobileNetV2 tabanli transfer learning modelini olusturur.
    
    Args:
        fine_tune (bool): True ise MobileNet'in son katmanlarini da egitir (daha yavas ama daha hassas olabilir).
    """
    print(">> Model olusturuluyor (MobileNetV2)...")

    # 1. Temel Model (Pre-trained Base)
    # include_top=False: Son siniflandirma katmanini at, biz kendi 10 sinifimizi ekleyecegiz.
    # weights='imagenet': ImageNet verisetiyle egitilmis agirliklari kullan.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    # Temel modeli dondur (Agirliklari degismesin)
    base_model.trainable = False

    if fine_tune:
        print("   -> Fine-tuning aktif: Son 30 katman egitilecek.")
        base_model.trainable = True
        # Ilk katmanlari dondur, sadece sonlari egit
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # 2. Yeni Katmanlari Ekleme
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Ozellik haritasini duz vektore cevir
        tf.keras.layers.Dropout(0.2),              # Overfitting'i onlemek icin %20 noronu kapat
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax') # 10 sinif icin olasilik ver
    ])

    # 3. Derleme (Compile)
    # Fine-tuning yapiliyorsa learning rate daha dusuk tutulmali
    lr = LEARNING_RATE / 10 if fine_tune else LEARNING_RATE
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy', # Etiketler integer oldugu icin 'sparse' kullaniyoruz
        metrics=['accuracy']
    )

    return model
