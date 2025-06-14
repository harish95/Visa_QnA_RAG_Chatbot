
# 🛂 Israel Entry QnA Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) based chatbot that helps users get accurate and up-to-date answers about entry rules for Israel. The system scrapes content from [https://israel-entry.piba.gov.il/](https://israel-entry.piba.gov.il/), converts it into semantic embeddings, and uses an LLM (LLaMA3 via Groq) to generate contextual answers with source references.

---

## 🚀 Features

- ✅ Live website content ingestion (WebBaseLoader)
- 🧠 High-quality dense retrieval using BGE or MiniLM embeddings
- 💬 Conversational chatbot with source documents shown
- 💾 Local persistent vectorstore using Chroma
- 🎨 User-friendly interface built in Streamlit
- 📜 Chat history with user + assistant messages

---

## 📁 Folder Structure

```
rag_qna/
├── build_vectorstore.py    # Script to scrape, embed, and save vector DB
├── app.py                  # Streamlit chatbot app
├── vectorstore/            # Saved Chroma DB files (auto-created)
├── .env                    # API key environment file
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/israel-entry-qna-chatbot.git
cd israel-entry-qna-chatbot
```

### 2. Create and Activate Environment

```bash
pip install -r requirements.txt
```

### 3. Add Your `.env` File

Create a `.env` file with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 4. Build the Vectorstore (Only Once)

```bash
python build_vectorstore.py
```

This will scrape the site and create embeddings saved locally in `vectorstore/`.

---

### 5. Run the Chatbot

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 🧠 Technologies Used

- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace Embeddings](https://huggingface.co/)
- [Groq LLaMA3](https://console.groq.com/)
- [Streamlit](https://streamlit.io/)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---

## 📌 Notes

- You can switch embedding models (e.g., `BAAI/bge-large-en-v1.5`, `all-mpnet-base-v2`) in both `build_vectorstore.py` and `app.py`.
- Chatbot answers include cited sources from the retrieved documents.
- Chat history is preserved in the Streamlit session.

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, extend, and contribute!

---

## 🙌 Acknowledgments

- [PIBA Israel Entry Website](https://israel-entry.piba.gov.il/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Groq for blazing-fast LLM inference](https://console.groq.com/)
