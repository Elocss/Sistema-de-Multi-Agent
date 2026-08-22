import os
import sys
from dotenv import load_dotenv

# Cargar variables de entorno del archivo .env en el directorio actual
load_dotenv()

# Validar que la API Key de OpenAI esté configurada
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY no está configurada en las variables de entorno ni en el archivo .env.", file=sys.stderr)
    print("Por favor, configúrala antes de ejecutar las consultas.", file=sys.stderr)

# Configurar el modelo por defecto. Usamos gpt-4o de OpenAI.
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

def get_llm(temperature=0.0, **kwargs):
    """Instancia y retorna el modelo ChatOpenAI con la configuración especificada."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        openai_api_key=OPENAI_API_KEY,
        **kwargs
    )

def get_embeddings():
    """Instancia y retorna el modelo de embeddings OpenAIEmbeddings."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )

def is_langfuse_configured():
    """Verifica si las credenciales de Langfuse están configuradas."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
