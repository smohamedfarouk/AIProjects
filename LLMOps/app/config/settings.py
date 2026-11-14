from dotenv import load_dotenv
import os

load_dotenv()

class Settings: 
    def __init__(self):
        self.APP_NAME = os.getenv("APP_NAME")
        self.APP_VERSION = os.getenv("APP_VERSION")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")
        self.PINECONE_REGION = os.getenv("PINECONE_REGION")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
        self.LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
        self.MYSQL_HOST = os.getenv("MYSQL_HOST")
        self.MYSQL_PORT = os.getenv("MYSQL_PORT")
        self.MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
        self.MYSQL_USER = os.getenv("MYSQL_USER")
        self.MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
        self.MYSQL_ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD")
        self.PDF_PATH = os.getenv("PDF_PATH")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.API_HOST = os.getenv("API_HOST", "127.0.0.1")
        self.API_PORT = os.getenv("API_PORT", "8000")

        self.ALLOWED_MODELS_NAMES = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    
settings = Settings()   