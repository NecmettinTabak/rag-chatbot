💬 FinanceRAG – Akbank Generative AI Bootcamp Projesi

🎯 Overview

FinanceRAG — bankacılık ve finans odaklı bir AI Chatbot projesidir.
Retrieval Augmented Generation (RAG) mimarisiyle geliştirilmiş bu uygulama, finansal verilerden anlamlı bilgi çıkararak kullanıcı sorularına akıllı, kaynak temelli yanıtlar üretir.

Uygulama, geleneksel chatbotların bilgi sınırlamasını ortadan kaldırarak, vektör veritabanı destekli bilgi getirme (retrieval) ve LLM tabanlı cevap üretme (generation) adımlarını birleştirir.

🧩 RAG Pipeline Akışı

Veri Yükleme: Hugging Face’ten finans/bankacılık veri setleri alınır

Metin Temizleme: soru–cevap cümleleri normalize edilir

Vektör Dönüşümü: sentence-transformers/all-MiniLM-L6-v2 embedding modeliyle semantik vektörler oluşturulur

Vektör Veritabanı: FAISS ile hızlı benzerlik araması yapılır

Sorgu Eşleştirme: Kullanıcının sorusu embedding’e dönüştürülür, en alakalı kayıtlar getirilir

Cevap Üretimi: Gemini API ile son cevap üretilir

📊 Dataset

📊 Kullanılan Veri Setleri

1️⃣ Bank Assistant QA Dataset
🔗 https://huggingface.co/datasets/wasifis/bank-assistant-qa

📝 Banka müşteri temsilcisi ile kullanıcı arasında geçen soru-cevap diyaloglarını içerir.
Finansal işlemler, kredi, hesap yönetimi ve destek senaryoları üzerine odaklıdır.

2️⃣ Bitext Retail Banking Chatbot Dataset
🔗 https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset

📝 Finans ve perakende bankacılığı için oluşturulmuş büyük ölçekli chatbot eğitimi veri setidir.
Gerçek müşteri taleplerine benzer örnek diyaloglar, intent ve response çiftleri içerir.

✨ Öne Çıkan Özellikler

💬 Gerçek Zamanlı Yanıtlar: LLM + FAISS entegrasyonu sayesinde hızlı cevap

🧠 Finansal Bilgi Zenginliği: Türkçe–İngilizce finans verileriyle eğitim

🔍 Kaynak Temelli Cevaplama: Her yanıt arka plandaki belgelere dayanır

🧰 Modüler Yapı: build_index.py ve rag_core.py ile pipeline tamamen yeniden oluşturulabilir

🔐 Güvenli Anahtar Yönetimi: .env dosyası ile API anahtarları izole edilir

⚙️ Kullanılan Teknolojiler
Katman	Teknoloji
Embedding	sentence-transformers/all-MiniLM-L6-v2
Vector DB	FAISS
Framework	LangChain, HuggingFace Datasets
LLM API	Gemini 1.5 Flash
Ortam	Python 3.13, PyCharm
Web Arayüzü (opsiyonel)	Streamlit
🧠 Kurulum ve Çalıştırma

⚙️ Kurulum ve Çalıştırma (Windows + PyCharm ortamı için)
1️⃣ Python 3.11 sanal ortamı oluştur

Projeni PyCharm veya PowerShell’de açtıktan sonra terminale şu komutu yaz:

python -m venv .venv311

2️⃣ Sanal ortamı etkinleştir

PowerShell veya VSCode terminalinde:

.venv311\Scripts\activate


<img width="612" height="42" alt="image" src="https://github.com/user-attachments/assets/8b52d3e6-10cb-4762-a0b2-a93fc84fbc40" />


Komut satırının başında yeşil (venv311) ibaresini gördüğünde aktif olmuştur ✅

3️⃣ Gereksinimleri yükle
pip install -r gereksinimler.txt


Tüm bağımlılıklar (langchain, faiss-cpu, datasets, transformers, tqdm, streamlit, vb.) yüklenecektir.

4️⃣ API anahtarını ekle

Proje kök dizininde .env dosyası oluştur ve içine:

GOOGLE_API_KEY=senin_api_keyin


ekle.

5️⃣ FAISS vektör indeksini oluştur
python uygulama/build_index.py


Bu adım, Hugging Face veri setlerinden embedding’leri çıkararak db/faiss_tr_finance/ altına kayıt yapar.

6️⃣ Chatbot’u çalıştır
python app/app.py


veya web arayüzü (Streamlit) olarak çalıştırmak istersen:

streamlit run app/app.py

<img width="867" height="170" alt="image" src="https://github.com/user-attachments/assets/794b25ab-49c3-4d29-aa5a-1880af65454f" />


📦 Proje Yapısı

rag-chatbot/
│
├── app/

│   ├── build_index.py
│   ├── rag_core.py
│   ├── app.py
│   └── db/faiss/
│

├── db/
├── gereksinimler.txt
├── .gitignore
└── README.md


