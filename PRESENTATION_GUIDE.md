# 🎤 Proje Sunum Rehberi

Hocaya sunum yaparken kullanabileceğin bir akış planı hazırladım. Bu maddeleri kendi cümlelerinle anlatman çok etkili olacaktır.

## 1. Giriş (Problem ve Amaç)
> "Hocam, projemin amacı uydu görüntülerini kullanarak yeryüzündeki alanların (orman, şehir, nehir vb.) otomatik olarak tespit edilmesidir."
- **Neden Önemli?** Bu sistem ormansızlaşmayı takip etmek, şehir planlaması yapmak veya tarım arazilerini izlemek için kullanılabilir.
- **Veri Seti:** EuroSAT veri setini kullandım. 10 farklı sınıf ve binlerce uydu görüntüsü içeriyor.

## 2. Yöntem (Kullandığım Teknolojiler)
> "Klasik yöntemler yerine, görüntü işleme konusunda en başarılı olan **Derin Öğrenme (Deep Learning)** yöntemini tercih ettim."
- **Model Mimarisi:** **MobileNetV2** kullandım.
  - *Neden?* Çünkü hem çok hızlı hem de başarısı kanıtlanmış bir model (Google tarafından geliştirildi).
  - **Transfer Learning:** Sıfırdan eğitmek yerine, önceden milyonlarca resimle eğitilmiş bir modelin "bilgisini" alıp kendi projemize uyarladım. Bu sayede çok daha yüksek doğruluk elde ettik.

## 3. Teknik Detaylar (Kodun İçindekiler)
Hoca teknik soru sorarsa şunları belirtebilirsin:
- **Veri Artırma (Data Augmentation):** Model resimleri ezberlemesin diye; eğitim sırasında resimleri rastgele çevirip, döndürüp, yakınlaştırarak veriyi çoğalttım.
- **Overfitting Önleme:** `Dropout` katmanı ekleyerek aşırı öğrenmenin önüne geçtim.
- **Optimizasyon:** `Adam` optimizasyon algoritmasını kullandım.

## 4. Canlı Demo (En Önemli Kısım!)
Burada Web Arayüzünü (`app.py`) açıp göstermelisin.
> "Sadece kod yazmakla kalmadım, bunu gerçek hayatta kullanılabilecek bir **Web Uygulamasına** dönüştürdüm."

1.  Terminale `streamlit run app.py` yazıp enter'a bas.
2.  Açılan sayfada bir uydu görüntüsü yükle.
3.  **"Analiz Et"** butonuna bas ve sonucun (olasılık grafiğiyle birlikte) nasıl geldiğini göster.

## 5. Sonuç
> "Sonuç olarak, geliştirdiğim model yüksek doğruluk oranıyla arazi tiplerini sınıflandırabiliyor ve hazırladığım arayüz sayesinde herkes tarafından kolayca kullanılabiliyor."

---
### 💡 İpucu
Hoca "Neyi daha iyi yapabilirdin?" diye sorarsa:
L"Daha fazla veriyle ve daha uzun süre (Epoch sayısını artırarak) eğitim yapsaydım model daha da hassas olabilirdi. Şu an 100 Epoch'a (senin yaptığın güncelleme) çıkardım ve sonuçları gözlemliyorum." diyebilirsin.
