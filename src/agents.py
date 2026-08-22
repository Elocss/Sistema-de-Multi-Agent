from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from src.config import get_llm
from src.vector_store import get_retriever_for_department

# 1. Definición del Esquema Estructurado para el Enrutador
class RoutingDecision(BaseModel):
    department: Literal["HR", "IT Support", "Finance", "Legal"] = Field(
        description="El departamento al que pertenece la consulta. HR para recursos humanos, vacaciones y licencias; IT Support para accesos de red, VPN y software; Finance para reembolsos de gastos y facturación; Legal para NDAs y contratos."
    )
    confidence: float = Field(
        description="Nivel de confianza en la clasificación entre 0.0 (nula) y 1.0 (absoluta)."
    )
    reasoning: str = Field(
        description="Breve explicación técnica detrás de la elección del departamento."
    )

def get_router_chain():
    """Retorna la cadena del Router que clasifica estructuradamente la consulta del usuario."""
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RoutingDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Eres un enrutador inteligente de consultas para una empresa SaaS.\n"
            "Tu tarea es analizar la consulta entrante del cliente y categorizarla en uno de los siguientes departamentos:\n"
            "- HR (Recursos Humanos, vacaciones, licencias, beneficios)\n"
            "- IT Support (Soporte técnico, accesos, VPN, software corporativo)\n"
            "- Finance (Finanzas, reembolso de gastos de viaje, facturación, nómina)\n"
            "- Legal (Acuerdos legales, NDAs, contratos de confidencialidad)\n\n"
            "Analiza detenidamente la intención y proporciona una respuesta estructurada."
        )),
        ("human", "{query}")
    ])
    
    return prompt | structured_llm

# 2. Configuración de los Agentes RAG Especializados utilizando LangGraph create_agent

def create_rag_agent(department: str):
    """Crea y retorna un CompiledStateGraph para el departamento específico con su respectivo RAG."""
    llm = get_llm(temperature=0.1)
    
    # Mapeo del nombre interno del directorio de Chroma al nombre del agente
    dept_map = {
        "HR": "hr",
        "IT Support": "it",
        "Finance": "finance",
        "Legal": "legal"
    }
    
    dept_key = dept_map.get(department, "hr")
    retriever = get_retriever_for_department(dept_key)
    
    # Crear herramienta de recuperación para el agente
    retriever_tool = create_retriever_tool(
        retriever,
        name=f"buscar_documentacion_{dept_key}",
        description=(
            f"Busca y recupera información oficial sobre las políticas y guías del departamento de {department}. "
            "Usa esta herramienta obligatoriamente para responder cualquier duda del cliente, "
            "asegurando que tus respuestas estén fundamentadas (grounded) en la documentación real de la empresa."
        )
    )
    
    tools = [retriever_tool]
    
    # Definir prompt del sistema del agente de dominio
    system_prompt = (
        f"Eres un agente de IA experto de soporte del departamento de {department} de la empresa.\n"
        "Tu objetivo es resolver las consultas de los clientes de manera precisa, profesional y empática.\n"
        "Instrucciones clave:\n"
        "1. Utiliza la herramienta de búsqueda de documentación para encontrar las políticas reales de la empresa.\n"
        "2. Responde en español.\n"
        "3. Basa tu respuesta exclusivamente en los fragmentos de documentación recuperados. "
        "Si no encuentras la respuesta en la documentación, di educadamente que no tienes esa información y que escalarás la consulta.\n"
        "4. Nunca inventes políticas, fechas o límites monetarios. Evita las alucinaciones."
    )
    
    # Crear el agente de LangGraph con el helper create_agent
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

# 3. Orquestador Principal
class SupportOrchestrator:
    def __init__(self):
        self.router_chain = get_router_chain()
        self.agents_cache = {}

    def get_agent_for_department(self, department: str):
        """Devuelve una instancia cacheada del agente de dominio."""
        if department not in self.agents_cache:
            self.agents_cache[department] = create_rag_agent(department)
        return self.agents_cache[department]

    async def route_and_resolve(self, query: str, callbacks: List = None) -> dict:
        """Enruta la consulta al agente RAG apropiado y retorna la respuesta."""
        # 1. Determinar departamento mediante el Router Chain
        routing_decision: RoutingDecision = await self.router_chain.ainvoke(
            {"query": query},
            config={"callbacks": callbacks}
        )
        
        department = routing_decision.department
        
        # 2. Obtener el agente de dominio especializado (CompiledStateGraph de LangGraph)
        agent = self.get_agent_for_department(department)
        
        # 3. Ejecutar el agente pasándole los mensajes
        agent_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"callbacks": callbacks}
        )
        
        # Extraer la respuesta del último mensaje en el historial del agente
        final_answer = agent_response["messages"][-1].content
        
        return {
            "query": query,
            "routed_department": department,
            "routing_confidence": routing_decision.confidence,
            "routing_reasoning": routing_decision.reasoning,
            "response": final_answer
        }
