import os
import uuid
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm, is_langfuse_configured
from src.vector_store import get_retriever_for_department

# 1. Esquema de Evaluación Estructurada
class EvaluationResult(BaseModel):
    relevance_score: int = Field(
        description="Puntuación de 1 a 5. ¿Qué tan bien responde la respuesta a la consulta original?"
    )
    relevance_feedback: str = Field(
        description="Justificación detallada de la puntuación de relevancia."
    )
    completeness_score: int = Field(
        description="Puntuación de 1 a 5. ¿Se responden todas las preguntas explícitas o implícitas de la consulta?"
    )
    completeness_feedback: str = Field(
        description="Justificación detallada de la puntuación de completitud."
    )
    accuracy_score: int = Field(
        description="Puntuación de 1 a 5. ¿La respuesta está estrictamente alineada con la documentación de soporte provista, evitando alucinaciones o datos no incluidos?"
    )
    accuracy_feedback: str = Field(
        description="Justificación detallada de la puntuación de precisión y grounding."
    )
    passed: bool = Field(
        description="Indica si la respuesta pasa los estándares mínimos (puntuación >= 3 en todas las dimensiones)."
    )
    action: Literal["approve", "retry", "escalate"] = Field(
        description="Recomendación final: 'approve' si es de alta calidad; 'retry' si falta contexto recuperable; 'escalate' si alucina o contradice las políticas."
    )

def get_evaluator_chain():
    """Retorna la cadena del Evaluador de calidad con salida estructurada."""
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(EvaluationResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Eres un auditor de calidad experto en sistemas conversacionales de IA corporativos.\n"
            "Tu tarea es evaluar la respuesta generada por un agente de soporte técnico basándote en la consulta del cliente y la documentación de referencia oficial.\n\n"
            "Dimensiones de Evaluación:\n"
            "1. RELEVANCIA: Que la respuesta aborde la consulta del cliente de manera directa sin desviarse.\n"
            "2. COMPLETITUD: Que la respuesta no deje preguntas sin responder.\n"
            "3. PRECISIÓN (GROUNDING): Que cada dato, límite monetario o procedimiento detallado esté estrictamente respaldado por los documentos provistos. Si la respuesta incluye información no presente en la documentación oficial, debe penalizarse severamente.\n\n"
            "Documentos de Referencia Oficiales:\n"
            "--- START DOCS ---\n"
            "{docs}\n"
            "--- END DOCS ---\n\n"
            "Consulta del Cliente: {query}\n"
            "Respuesta a Evaluar: {response}"
        )),
        ("human", "Realiza la evaluación de la respuesta y proporciona el JSON estructurado con los puntajes, justificaciones y tu recomendación de acción.")
    ])
    
    return prompt | structured_llm

# 2. Función Principal de Evaluación e Integración con Langfuse
async def run_evaluation(
    query: str,
    response: str,
    department: str,
    trace_id: Optional[str] = None
) -> dict:
    """Evalúa la respuesta generada, recupera la documentación de referencia y opcionalmente reporta a Langfuse."""
    
    # 1. Recuperar la documentación relevante para la evaluación
    dept_map = {
        "HR": "hr",
        "IT Support": "it",
        "Finance": "finance",
        "Legal": "legal"
    }
    dept_key = dept_map.get(department, "hr")
    retriever = get_retriever_for_department(dept_key)
    docs = await retriever.ainvoke(query)
    
    docs_context = "\n\n".join([
        f"Archivo: {doc.metadata.get('filename')}\nContenido:\n{doc.page_content}"
        for doc in docs
    ])
    
    # 2. Ejecutar la evaluación con el LLM
    evaluator_chain = get_evaluator_chain()
    evaluation_result: EvaluationResult = await evaluator_chain.ainvoke({
        "docs": docs_context,
        "query": query,
        "response": response
    })
    
    result_dict = evaluation_result.model_dump()
    
    # 3. Reportar puntuaciones a Langfuse si está configurado y se provee trace_id
    if trace_id and is_langfuse_configured():
        try:
            from langfuse import Langfuse
            langfuse_client = Langfuse(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            )
            
            # Registrar score de Relevancia
            langfuse_client.score(
                name="relevance",
                value=float(result_dict["relevance_score"]),
                comment=result_dict["relevance_feedback"],
                trace_id=trace_id
            )
            
            # Registrar score de Completitud
            langfuse_client.score(
                name="completeness",
                value=float(result_dict["completeness_score"]),
                comment=result_dict["completeness_feedback"],
                trace_id=trace_id
            )
            
            # Registrar score de Precisión
            langfuse_client.score(
                name="accuracy",
                value=float(result_dict["accuracy_score"]),
                comment=result_dict["accuracy_feedback"],
                trace_id=trace_id
            )
            
            print(f"Métricas de evaluación subidas con éxito a Langfuse para el Trace ID: {trace_id}")
            
        except Exception as e:
            print(f"Advertencia: Error al subir puntuaciones a Langfuse: {e}")
            
    return result_dict
