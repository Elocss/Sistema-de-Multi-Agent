import asyncio
import os
import sys

# Agregar el directorio raíz al path para resolver importaciones del paquete 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vector_store import initialize_vector_stores
from src.pipeline import SupportPipeline

async def main():
    print("==================================================")
    print("Sistema Multi-Agente de Soporte con LangChain y Langfuse")
    print("==================================================")
    
    # 1. Inicializar bases de datos vectoriales
    print("\n[Inicialización] Verificando base de datos vectorial...")
    try:
        initialize_vector_stores()
        print("[Inicialización] Almacenes vectoriales listos.")
    except Exception as e:
        print(f"[Inicialización] Error al inicializar bases vectoriales: {e}", file=sys.stderr)
        return

    # 2. Instanciar pipeline
    pipeline = SupportPipeline()
    
    # 3. Bucle interactivo
    print("\nEscribe tu consulta de soporte (o escribe 'salir' para terminar):")
    while True:
        try:
            query = input("\nCliente > ").strip()
            if not query:
                continue
            if query.lower() in ["salir", "exit", "quit"]:
                print("Saliendo del sistema de soporte...")
                break
                
            # Ejecutar consulta
            result = await pipeline.run(query)
            
            # Mostrar resultados
            print("\n---------------- RESULTADO DEL SISTEMA ----------------")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Trace ID: {result['trace_id']}")
                print(f"Departamento Enrutado: {result['routed_department']} (Confianza: {result['routing_confidence']:.2f})")
                print(f"Razonamiento Ruteo: {result['routing_reasoning']}")
                print(f"\nRespuesta final del Agente:\n{result['response']}")
                print(f"\nEstado Calidad: {result['status'].upper()}")
                
            print("-------------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nSaliendo del sistema de soporte...")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")

if __name__ == "__main__":
    # Windows event loop policy support for async
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
