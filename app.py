import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# ==========================================
# AYARLAR VE SABITLER
# ==========================================
st.set_page_config(
    page_title="Uydu Görüntü Analizi",
    page_icon="🛰️",
    layout="centered"
)

# Sinif isimleri (Model egitim sirasina gore)
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# ==========================================
# CSS TASARIM (MODERN GORUNUM)
# ==========================================
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .upload-text {
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        color: #333;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .confidence-score {
        font-size: 24px;
        color: #2e7d32;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# MODEL YUKLEME (CACHE)
# ==========================================
@st.cache_resource
def load_model():
    """Modeli bir kez yukler ve cache'ler."""
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

# ==========================================
# YARDIMCI FONKSIYONLAR
# ==========================================
def preprocess_image(image):
    """Resmi modelin bekledigi formata getirir (64x64, normalize)."""
    image = image.resize((64, 64))
    image_array = np.array(image)
    
    # Eger resim RGBA ise RGB'ye cevir
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
        
    image_array = image_array / 255.0  # Normalize et [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # (1, 64, 64, 3)
    return image_array

# ==========================================
# ANA UYGULAMA
# ==========================================
def main():
    # Baslik ve Aciklama
    st.title("🛰️ Uydu Görüntü Sınıflandırma")
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>Bu yapay zeka modeli, yüklediğiniz uydu fotoğraflarını analiz ederek <b>Arazi Tipini</b> tespit eder.</p>", unsafe_allow_html=True)

    # Model Hazirlik
    with st.spinner('Yapay Zeka Modeli Yükleniyor...'):
        model = load_model()

    if model is None:
        st.warning("Lütfen önce modeli eğitin: `python main.py train`")
        return

    # Resim Yukleme Alani
    st.markdown("<div class='upload-text'>Lütfen Analiz Edilecek Resmi Yükleyin</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Resmi Goster
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
            
            # Analiz Butonu
            if st.button('GÖRÜNTÜYÜ ANALİZ ET'):
                with st.spinner('Görüntü İşleniyor...'):
                    # Simule edilmis bekleme suresi (etki icin)
                    time.sleep(1)
                    
                    # Tahmin
                    processed_img = preprocess_image(image)
                    predictions = model.predict(processed_img)
                    score = tf.nn.softmax(predictions[0])
                    
                    predicted_class = CLASS_NAMES[np.argmax(score)]
                    confidence = 100 * np.max(score)

                # Sonuclari Goster
                st.markdown(f"""
                <div class='prediction-box'>
                    <h3>Analiz Sonucu:</h3>
                    <h1 style='color: #1565C0;'>{predicted_class}</h1>
                    <p>Doğruluk Payı: <span class='confidence-score'>%{confidence:.2f}</span></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar Chart ile Olasiliklar
                st.markdown("### Detaylı Olasılık Dağılımı")
                st.bar_chart({name: float(pred) for name, pred in zip(CLASS_NAMES, predictions[0])})
                
                if confidence > 90:
                    st.success("Tebrikler! Model bu sonuçtan oldukça emin. ✅")
                elif confidence > 70:
                    st.info("Model sonuçtan emin görünüyor. ℹ️")
                else:
                    st.warning("Model tam karar veremedi, resim net olmayabilir. ⚠️")
                    
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

if __name__ == "__main__":
    main()
