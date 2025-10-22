import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os



# .env dosyasını proje kökünden yükle
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ✅ Ortam değişkenlerini yükle
load_dotenv()

# ✅ API anahtarını .env’den çek
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 📁 FAISS dizini
DB_DIR = os.path.join(os.path.dirname(__file__), "db", "faiss")

# 🔢 Embedding modeli
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🧠 FAISS veritabanını yükle
db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

# 🚀 Gemini modelini başlat
model = genai.GenerativeModel("gemini-2.5-flash")

def ask_gemini(question: str):
    """FAISS + Gemini tabanlı akıllı cevaplama"""
    try:
        # 🔹 FAISS'ten ilgili dokümanları getir
        docs = db.similarity_search(question, k=5)
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
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Bir hata oluştu: {e}"
