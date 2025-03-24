from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from src.llm_client import LlmClient
from src.utils.utils import BLUE, PURPLE, RESET, SCISSORS, PASTEL_YELLOW, get_env_key
import os
import streamlit as st

class RAG:
    _instance = None

    @classmethod
    def get_instance(cls, data_dir="context", collection_name="rag_index"):
        if not cls._instance:
            cls._instance = cls.__new__(cls)
            cls._instance._initialized = False
            cls._instance.__init__(data_dir, collection_name)
        return cls._instance

    def __init__(self, data_dir="context", collection_name="rag_index"):
        if self._initialized:
            return
        self._initialized = True

        self.data_dir = data_dir
        self.collection_name = collection_name
        self.docs = []

        self.qdrant_client = QdrantClient(
            url=get_env_key("QDRANT_URL"),
            api_key=get_env_key("QDRANT_API_KEY")
        )

        if "qdrant_index_loaded" not in st.session_state:
            st.session_state.qdrant_index_loaded = False
            st.session_state.db = None
            st.session_state.retriever = None

        if not st.session_state.qdrant_index_loaded:
            self._load_or_create_index()
        else:
            print("✅ Colección ya existente.")
            self.db = st.session_state.db
            self.retriever = st.session_state.retriever

        self.llm_client = LlmClient.get_instance()

    def _load_or_create_index(self):
        if st.session_state.get("qdrant_index_loaded", False):
            print("🟡 Índice ya cargado, omitiendo carga adicional.")
            return

        print(f"{BLUE}📁 Cargando documentos desde '{self.data_dir}'...{RESET}")
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            if file_name.endswith(".txt"):
                print(f"📝 TXT → {file_name}")
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_name.endswith(".pdf"):
                print(f"📄 PDF → {file_name}")
                loader = PyPDFLoader(file_path)
            else:
                print(f"{PASTEL_YELLOW}📜 Formato no soportado:{RESET} {file_name}")
                continue
            self.docs.extend(loader.load())

        print(f"{SCISSORS} {PASTEL_YELLOW} Dividiendo en chunks...{RESET}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.docs = text_splitter.split_documents(self.docs)

        print("🤖 Generando embeddings...")
        model_path = "sentence-transformers/all-mpnet-base-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            # model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            model_kwargs={},  # Use CPU
            encode_kwargs={'normalize_embeddings': False},
        )

        if self.collection_name not in [col.name for col in self.qdrant_client.get_collections().collections]:
            print("🧠 Creando colección en Qdrant Cloud...")
            self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        print("📦 Subiendo documentos a Qdrant...")
        self.db = Qdrant.from_documents(
            self.docs,
            embeddings,
            url=get_env_key("QDRANT_URL"),
            api_key=get_env_key("QDRANT_API_KEY"),
            collection_name=self.collection_name,
        )

        self.retriever = self.db.as_retriever(search_kwargs={"k": 4})
        st.session_state.db = self.db
        st.session_state.retriever = self.retriever
        st.session_state.qdrant_index_loaded = True
        print(f"🎉 {PURPLE}Carga finalizada con {len(self.docs)} fragmentos.{RESET}")
        print(f"🗂️{PURPLE} Qdrant creado con {len(self.docs)} documentos.{RESET}")

    def chat(self):
        print("Bienvenido al chat RAG. Escribe 'salir' para terminar la conversación.")
        conversation_history = []

        while True:
            question = input("Tú: ")
            if question.lower() in ["salir", "exit", "quit"]:
                print("Finalizando la sesión...")
                break

            results = self.retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in results])

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "system", "content": f"Contexto proporcionado:\n{context}"})

            response = self.llm_client.get_response(conversation_history)
            print(f"RAG: {response}")

    def ask_question(self, question):
        results = self.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in results])
        messages_with_context = [{"role": "user", "content": f"{question}\n\nContexto proporcionado:\n{context}"}]
        response = self.llm_client.get_response(messages_with_context)
        return response

if __name__ == "__main__":
    rag_chat = RAG.get_instance()
    response = rag_chat.ask_question("¿En qué consiste la pirámide invertida de 6 cartas?")
    print("Respuesta:", response)
