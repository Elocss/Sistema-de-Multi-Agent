import os
import uuid
import asyncio
from typing import Optional
from src.config import is_langfuse_configured
from src.agents import SupportOrchestrator
from src.evaluator import run_evaluation

class SupportPipeline:
    def __init__(self):
        self.orchestrator = SupportOrchestrator()

    async def run(self, query: str, max_retries: int = 1) -> dict:
        """Ejecuta el flujo completo de soporte: ruteo, RAG, evaluación y auto-corrección."""
        
        # 1. Generar un trace_id único para Langfuse
        trace_id = str(uuid.uuid4())
        
        # 2. Configurar el callback de Langfuse si está habilitado
        callbacks = []
        if is_langfuse_configured():
            try:
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                    trace_id=trace_id
                )
                callbacks.append(langfuse_handler)
                print(f"Tracer de Langfuse activo. Trace ID: {trace_id}")
            except Exception as e:
                print(f"Advertencia: No se pudo inicializar el callback de Langfuse: {e}")
        else:
            print("Langfuse no está configurado. La ejecución continuará localmente sin tracking remoto.")
            
        current_query = query
        retries_left = max_retries
        last_response = None
        routed_dept = None
        routing_confidence = 0.0
        routing_reasoning = ""
        
        while retries_left >= 0:
            # 3. Enrutar y Resolver
            print(f"\n[Pipeline] Procesando consulta (Intentos restantes: {retries_left})...")
            try:
                resolution = await self.orchestrator.route_and_resolve(current_query, callbacks=callbacks)
                routed_dept = resolution["routed_department"]
                routing_confidence = resolution["routing_confidence"]
                routing_reasoning = resolution["routing_reasoning"]
                last_response = resolution["response"]
            except Exception as e:
                print(f"Error durante el enrutamiento/resolución: {e}")
                return {
                    "trace_id": trace_id,
                    "query": query,
                    "error": str(e),
                    "status": "failed"
                }

            # 4. Evaluación Automatizada de Calidad (LLM-as-a-judge)
            print("[Pipeline] Evaluando respuesta generada...")
            eval_result = await run_evaluation(
                query=query,
                response=last_response,
                department=routed_dept,
                trace_id=trace_id
            )
            
            print(f"[Evaluación] Passed: {eval_result['passed']} | Action: {eval_result['action']}")
            print(f"   - Relevancia: {eval_result['relevance_score']}/5: {eval_result['relevance_feedback']}")
            print(f"   - Completitud: {eval_result['completeness_score']}/5: {eval_result['completeness_feedback']}")
            print(f"   - Precisión (Grounding): {eval_result['accuracy_score']}/5: {eval_result['accuracy_feedback']}")

            # Si pasa la evaluación o no quedan reintentos, terminamos el flujo
            if eval_result["passed"] or eval_result["action"] != "retry" or retries_left == 0:
                return {
                    "trace_id": trace_id,
                    "query": query,
                    "routed_department": routed_dept,
                    "routing_confidence": routing_confidence,
                    "routing_reasoning": routing_reasoning,
                    "response": last_response,
                    "evaluation": eval_result,
                    "retries_performed": max_retries - retries_left,
                    "status": "success" if eval_result["passed"] else "flagged"
                }
            
            # 5. Lógica de auto-corrección para el reintento
            print("\n[Pipeline] La respuesta no superó el umbral de calidad. Aplicando auto-corrección...")
            feedback = (
                f"La respuesta anterior fue rechazada por control de calidad con el siguiente feedback:\n"
                f"- Relevancia: {eval_result['relevance_feedback']}\n"
                f"- Completitud: {eval_result['completeness_feedback']}\n"
                f"- Precisión: {eval_result['accuracy_feedback']}\n\n"
                f"Por favor, reformula tu respuesta para corregir estos problemas. "
                f"Asegúrate de basarte ÚNICAMENTE en la documentación provista."
            )
            
            # Concatenamos el feedback a la consulta para el agente
            current_query = f"{query}\n\n[SISTEMA DE CALIDAD - RETRY FEEDBACK]\n{feedback}"
            retries_left -= 1
            
        return {
            "trace_id": trace_id,
            "query": query,
            "routed_department": routed_dept,
            "routing_confidence": routing_confidence,
            "routing_reasoning": routing_reasoning,
            "response": last_response,
            "evaluation": eval_result,
            "retries_performed": max_retries,
            "status": "flagged"
        }
