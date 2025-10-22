import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Kullanılacak datasetler
DATASETS = [
    "wasifis/bank-assistant-qa",
    "bitext/Bitext-retail-banking-llm-chatbot-training-dataset"
]


DB_DIR = "db/faiss"

def load_texts_from_datasets(dataset_names):
    texts = []
    for name in dataset_names:
        print(f"📥 {name} yükleniyor...")
        ds = load_dataset(name)

        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        count = 0

        for row in split:
            q = (
                row.get("question")
                or row.get("query")
                or row.get("user_input")
                or row.get("instruction")
                or row.get("input")
                or row.get("text")
                or ""
            )
            a = (
                row.get("answer")
                or row.get("response")
                or row.get("output")
                or row.get("output_text")
                or ""
            )

            if q and a:
                texts.append(f"Soru: {q}\nCevap: {a}")
                count += 1

        print(f"✅ {name} datasetinden {count} kayıt işlendi.")

    print(f"📊 Toplam {len(texts)} metin birleşti.")
    return texts


def main():
    os.makedirs(DB_DIR, exist_ok=True)
    texts = load_texts_from_datasets(DATASETS)

    if not texts:
        print("❌ Hiç metin bulunamadı, alan adlarını kontrol et!")
        return

    print("✂ Metinler bölünüyor...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents(texts)

    print("🔢 Embedding’ler oluşturuluyor...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("💾 FAISS veritabanına kaydediliyor...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_DIR)

    print(f"✅ FAISS index başarıyla oluşturuldu: {DB_DIR}")


if __name__ == "_main_":
    main()