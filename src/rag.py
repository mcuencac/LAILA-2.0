import os
from dotenv import load_dotenv
load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from src.llm_client import LlmClient
import torch
from src.utils.utils import BLUE, PURPLE, RESET, SCISSORS, PASTEL_YELLOW, get_env_key
from tqdm import tqdm

class RAG:
    _instance = None  # Singleton

    @classmethod
    def get_instance(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, data_dir="context", collection_name="laila_collection"):
        self.collection_name = collection_name
        self.docs = []
        self.data_dir = data_dir
        self.llm_client = LlmClient.get_instance()
        self._load_or_create_index(data_dir)

    def _load_or_create_index(self, data_dir):
        client = QdrantClient(
            url="https://093c987e-86ec-46bb-9112-2ada146b9794.eu-west-1-0.aws.cloud.qdrant.io",
            api_key=os.getenv("QDRANT_API_KEY")
        )

        if not client.collection_exists(self.collection_name):    
            print("🧠 Creando colección en Qdrant Cloud...")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

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
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            self.docs = self.text_splitter.split_documents(self.docs)

            print("🤖 Generando embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
            )

            self.db = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

            batch_size = 10
            total = len(self.docs)
            print("📦 Subiendo documentos a Qdrant...")
            for i in tqdm(range(0, total, batch_size), desc="Progreso", unit="batch"):
                batch = self.docs[i:i+batch_size]
                self.db.add_documents(batch)

            print(f"🎉 {PURPLE}Carga finalizada con {len(self.docs)} fragmentos.{RESET}")
            print(f"📱 {PURPLE} Qdrant creado con {len(self.docs)} documentos.{RESET}")

        else:
            print("🟡 Índice ya cargado, omitiendo carga adicional.")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # ✅ Actualizado a MiniLM
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': False},
            )
            print("🤖 Usamos los mismos embeddings.")
            self.db = QdrantVectorStore(
                client=client, 
                collection_name=self.collection_name,
                embedding=self.embeddings, # Se usa el mismo embedding
            )
        
        self.retriever = self.db.as_retriever(search_kwargs={"k": 4}) # Retriever con k=4

    def ask_question(self, question):
        results = self.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in results])
        messages_with_context = [{"role": "user", "content": f"{question}\n\nContexto:\n{context}"}]
        return self.llm_client.get_response(messages_with_context)


# if __name__ == "__main__":
#     rag = RAG()
#     question = input("🔮 Introduce tu pregunta: ")
#     response = rag.ask_question(question)
#     print("\n📜 Respuesta del sistema:\n")
#     print(response)