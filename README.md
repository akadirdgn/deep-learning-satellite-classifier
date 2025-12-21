# 🛰️ Satellite Image Classification using Deep Learning

Bu proje, uydu görüntülerini kullanarak arazi tiplerini (orman, nehir, otoyol vb.) sınıflandırmak için geliştirilmiş bir Derin Öğrenme (Deep Learning) modelidir.

## 📋 1. Problem Tanımı ve Amaç
EuroSAT veri seti kullanılarak uydu görüntülerinin otomatik olarak etiketlenmesi hedeflenmiştir. Bu sistem, ormansızlaşma takibi, şehir planlama ve çevre analizi gibi alanlarda kullanılabilir.
- **Giriş:** 64x64 piksel RGB uydu görüntüleri.
- **Çıkış:** 10 farklı arazi sınıfı (AnnualCrop, Forest, Highway, vb.).
- **Hedef:** Yüksek doğruluk oranı ile sınıflandırma yapmak.

## 🛠️ 2. Kullanılan Yöntemler (Methodology)

### Veri Seti ve Ön İşleme
- **Veri Seti:** EuroSAT (Land Use and Land Cover Classification).
- **Ön İşleme:**
  - Görüntüler 64x64 boyutuna yeniden boyutlandırıldı.
  - Piksel değerleri [0, 1] aralığına normalize edildi.
  - Eğitim (%70), Doğrulama (%15) ve Test (%15) olarak ayrıldı.

### Model Mimarisi: Transfer Learning
Projede **MobileNetV2** mimarisi kullanılmıştır.
- **Neden MobileNetV2?** Hafif, hızlı ve mobil/web uygulamaları için optimize edilmiştir.
- **Transfer Learning:** ImageNet üzerinde eğitilmiş ağırlıklar kullanılarak eğitim süresi kısaltılmış ve başarı artırılmıştır.
- **Ek Katmanlar:**
  - `GlobalAveragePooling2D`: Özellik haritasını vektöre çevirmek için.
  - `Dropout (0.2)`: Overfitting'i (aşırı öğrenme) önlemek için.
  - `Dense (Softmax)`: 10 sınıf için olasılık dağılımı.

### Veri Artırma (Data Augmentation)
Overfitting'i azaltmak için eğitim sırasında rastgele dönüşümler uygulanmıştır:
- `RandomFlip`: Yatay çevirme.
- `RandomRotation`: Döndürme (%20).
- `RandomZoom`: Yakınlaştırma (%20).

## 📊 3. Deneysel Sonuçlar
Model 5 Epoch boyunca eğitilmiş ve aşağıdaki metriklerle değerlendirilmiştir:
- **Loss Function:** Sparse Categorical Crossentropy.
- **Optimizer:** Adam (Learning Rate: 0.001).
- **Metric:** Accuracy (Doğruluk).

*(Buraya eğitim sonucunda elde edilen Accuracy ve Loss grafikleri eklenebilir)*

## 🚀 4. Kurulum ve Kullanım

### Gereksinimler
```bash
pip install -r requirements.txt
```

### Modeli Eğitme
```bash
python main.py train
```

### Arayüzü Başlatma (Streamlit)
Web tabanlı arayüz üzerinden kendi resimlerinizi test etmek için:
```bash
streamlit run app.py
```

## 📁 Proje Yapısı
- `scripts/`: Eğitim ve değerlendirme kodları.
- `models/`: Model mimarisi tanımları.
- `data/`: Veri yükleme ve işleme fonksiyonları.
- `app.py`: Streamlit web arayüzü.

## 👥 Katkıda Bulunanlar
- **Kadir Doğan** - Geliştirici
