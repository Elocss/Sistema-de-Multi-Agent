import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from src.config import get_embeddings

# Directorios de documentos y base de datos vectorial
DOCS_DIR = "company_docs"
CHROMA_PERSIST_DIR = "data/chroma"

DEPARTMENTS = ["hr", "it", "finance", "legal"]

def load_documents_for_department(department: str) -> list[Document]:
    """Lee todos los archivos markdown en la carpeta del departamento y retorna documentos."""
    dept_dir = os.path.join(DOCS_DIR, department)
    documents = []
    
    if not os.path.exists(dept_dir):
        print(f"Directorio no encontrado para {department}: {dept_dir}")
        return documents
        
    for filename in os.listdir(dept_dir):
        if filename.endswith(".md") or filename.endswith(".txt"):
            filepath = os.path.join(dept_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                # Crear objeto Document de LangChain
                doc = Document(
                    page_content=content,
                    metadata={"source": filepath, "filename": filename, "department": department}
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error al leer archivo {filepath}: {e}")
                
    return documents

def initialize_vector_stores(force_recreate: bool = False):
    """Inicializa e indexa los almacenes de vectores para todos los departamentos."""
    embeddings = get_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    for dept in DEPARTMENTS:
        persist_path = os.path.join(CHROMA_PERSIST_DIR, dept)
        
        # Si ya existe y no forzamos recreación, saltamos la indexación
        if os.path.exists(persist_path) and not force_recreate:
            if len(os.listdir(persist_path)) > 0:
                # Almacén ya inicializado
                continue
                
        print(f"Indexando documentos para el departamento: {dept.upper()}...")
        
        # Cargar y dividir documentos
        raw_docs = load_documents_for_department(dept)
        if not raw_docs:
            print(f"Advertencia: No hay documentos para indexar en '{dept}'.")
            # Crear un documento vacío de respaldo para que Chroma no falle por falta de documentos
            raw_docs = [Document(page_content=f"Documentación vacía de {dept}.", metadata={"source": "dummy"})]
            
        split_docs = text_splitter.split_documents(raw_docs)
        
        # Crear e indexar en Chroma
        Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_path
        )
        print(f"Departamento {dept.upper()} indexado correctamente en {persist_path}.")

def get_retriever_for_department(department: str):
    """Retorna el retriever de Chroma correspondiente al departamento especificado."""
    # Asegurarnos de que los almacenes de vectores estén inicializados
    initialize_vector_stores()
    
    embeddings = get_embeddings()
    persist_path = os.path.join(CHROMA_PERSIST_DIR, department.lower())
    
    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embeddings
    )
    
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
