import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env en el directorio actual
load_dotenv()

# Validar que la API Key de Gemini esté configurada
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY no está configurada en las variables de entorno ni en el archivo .env.", file=sys.stderr)
    print("Por favor, configúrala antes de ejecutar las consultas.", file=sys.stderr)

# Configurar el modelo por defecto. Usamos gemini-1.5-flash o gemini-2.0-flash por estabilidad y velocidad.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDINGS_MODEL", "models/text-embedding-004")

def get_llm(temperature=0.0, **kwargs):
    """Instancia y retorna el modelo ChatGoogleGenAI con la configuración especificada."""
    from langchain_google_genai import ChatGoogleGenAI
    return ChatGoogleGenAI(
        model=MODEL_NAME,
        temperature=temperature,
        google_api_key=GEMINI_API_KEY,
        **kwargs
    )

def get_embeddings():
    """Instancia y retorna el modelo de embeddings GoogleGenAIEmbeddings."""
    from langchain_google_genai import GoogleGenAIEmbeddings
    return GoogleGenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GEMINI_API_KEY
    )

def is_langfuse_configured():
    """Verifica si las credenciales de Langfuse están configuradas."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
