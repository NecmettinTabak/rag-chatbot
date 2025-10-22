import streamlit as st
from rag_core import ask_gemini

# 🧭 Sayfa ayarları
st.set_page_config(page_title="💬 Finans Asistan Chatbot", page_icon="💳", layout="wide")


# 🏦 Başlık ve açıklama
st.title("💳 Finans Asistan Chatbot")
st.markdown("Akbank benzeri bir bankacılık destek asistanına hoş geldiniz!")

# 🧠 Oturumda model hazır mı kontrol et
if "model_ready" not in st.session_state:
    with st.spinner("Model yükleniyor..."):
        st.session_state.model_ready = True
        st.success("Model başarıyla yüklendi ✅")

# 🗣️ Kullanıcı sorusu
question = st.text_input(
    "Sorunuzu yazın (örnek: *Kredi kartı başvurusu nasıl yapılır?*)",
    placeholder="Kredi kartı borcumu nasıl öğrenebilirim?",
)

# 💬 Yanıt kısmı
if question:
    with st.spinner("Yanıt hazırlanıyor..."):
        try:
            answer = ask_gemini(question)
            st.markdown("### 💬 Cevap:")
            st.write(answer)
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

# 📌 Alt bilgi
st.markdown(
    """
    ---
    💡 **Finans Asistan Chatbot**, bankacılık işlemleri, kredi, kart başvurusu ve müşteri destek konularında
    hızlı bilgi sağlamak amacıyla geliştirilmiştir.
    """
)
