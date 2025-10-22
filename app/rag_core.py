import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset  # ✅ FAISS yeniden inşa için gerekli


# ✅ Ortam değişkenlerini yükle
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 📁 FAISS dizini
DB_DIR = os.path.join(os.path.dirname(__file__), "db", "faiss")

# 🔢 Embedding modeli
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Veri setleri (FAISS yeniden oluşturma durumunda kullanılacak)
DATASETS = [
    "wasifis/bank-assistant-qa",
    "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
]


def build_faiss_index():
    """Cloud ortamında FAISS index yoksa otomatik oluşturur."""
    try:
        texts = []
        for name in DATASETS:
            print(f"📥 {name} yükleniyor...")
            ds = load_dataset(name)
            split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

            for row in split:
                q = row.get("question") or row.get("query") or row.get("user_input") or ""
                a = row.get("answer") or row.get("response") or ""
                if q and a:
                    texts.append(f"Soru: {q}\nCevap: {a}")

        print("🧩 Embedding ve FAISS index oluşturuluyor...")
        db_local = FAISS.from_texts(texts, embeddings)
        os.makedirs(DB_DIR, exist_ok=True)
        db_local.save_local(DB_DIR)
        print("✅ FAISS index başarıyla oluşturuldu.")
        return db_local
    except Exception as e:
        print(f"❌ FAISS index oluşturulamadı: {e}")
        return None


# 🧠 FAISS veritabanını yükle veya oluştur
try:
    db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index yüklendi.")
except Exception:
    print("⚠️ FAISS index bulunamadı, yeniden oluşturuluyor...")
    db = build_faiss_index()

# 🚀 Gemini modelini başlat
model = genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(question: str):
    """FAISS + Gemini tabanlı akıllı cevaplama"""
    try:
        if db is None:
            return "FAISS veritabanı oluşturulamadı, lütfen yeniden başlatın."

        # 🔹 FAISS'ten ilgili dokümanları getir
        docs = db.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 🔹 Prompt oluştur
        prompt = f"""
        Sen bir bankacılık destek asistanısın.
        Aşağıdaki bağlamı kullanarak kullanıcının sorusuna net ve doğru bir yanıt ver.

        Bağlam:
        {context}

        Soru:
        {question}

        Cevap:
        """

        # 🔹 Gemini'den yanıt al
        response = model.generate_content(prompt, request_options={"timeout": 30})

        return response.text

    except Exception as e:
        return f"Bir hata oluştu: {e}"
