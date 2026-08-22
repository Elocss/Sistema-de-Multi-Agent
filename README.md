# Sistema de Multi-Agent Support con Ruteo Inteligente, RAG y Observabilidad

Este proyecto es un sistema de soporte al cliente multi-agente de grado de producción diseñado para clasificar consultas entrantes por departamento (HR, IT Support, Finance, Legal) y dirigirlas a agentes RAG especializados que generan respuestas precisas utilizando la documentación interna de la empresa. Todo el sistema está instrumentado para el seguimiento de la ejecución con **LangChain** y **Langfuse**.

## Características del Sistema

1. **Ruteo Inteligente**: Un agente clasificador principal analiza la consulta del cliente y devuelve una respuesta estructurada en formato JSON (`RoutingDecision`), indicando el departamento correspondiente, la confianza de la predicción y el razonamiento.
2. **Agentes RAG Especializados**: 4 agentes de dominio diferentes que utilizan bases de datos vectoriales locales **Chroma** independientes para cada área (`HR`, `IT Support`, `Finance`, `Legal`). Cada agente está confinado a buscar solo en su área documental.
3. **Evaluación de Calidad Automatizada**: Un agente evaluador "LLM-as-a-judge" revisa la respuesta final del agente comparándola con la consulta original y los documentos oficiales en 3 dimensiones de calidad: **Relevancia**, **Completitud** y **Precisión/Groundedness**.
4. **Auto-Corrección Activa**: Si una respuesta es rechazada en la evaluación de calidad, el pipeline retroalimenta de forma automática al agente con los comentarios del evaluador para que corrija la respuesta en un reintento.
5. **Observabilidad en Producción**: Trazabilidad completa integrada con **Langfuse**, lo que permite rastrear cada paso de la cadena (el ruteo, las llamadas al recuperador de Chroma, la generación del LLM y las puntuaciones de calidad del evaluador).

---

## Decisiones Técnicas

- **LangChain**: Elegido por su robustez al encapsular el flujo mediante cadenas LCEL y la facilidad para usar salidas estructuradas (`.with_structured_output`), lo que garantiza que las clasificaciones y evaluaciones mantengan un formato JSON confiable.
- **Chroma**: Base de datos vectorial persistente e incrustada en Python que permite segmentar de manera eficiente las colecciones por dominio documental para evitar que el agente de IT lea documentos financieros o viceversa.
- **Langfuse**: Utilizado en lugar de logs de consola estándar porque permite realizar un seguimiento jerárquico de las trazas (el ruteo y el agente RAG pertenecen a una misma transacción/trace_id), facilitando enormemente la auditoría y depuración en producción de respuestas incorrectas.

---

## Estructura de Directorios

```text
Sistema-de-Multi-Agent/
├── company_docs/            # Documentos de referencia para RAG
│   ├── hr/                  # Recursos Humanos (Vacaciones)
│   ├── it/                  # Soporte IT (VPN)
│   ├── finance/             # Finanzas (Gastos de viaje)
│   └── legal/               # Legal (Firma de NDA)
├── src/
│   ├── agents.py            # Enrutador, agentes RAG y orquestador
│   ├── config.py            # Carga de variables de entorno y clientes LLM
│   ├── evaluator.py         # LLM-as-a-judge evaluador e integración de scores
│   ├── main.py              # Aplicación interactiva de consola
│   ├── pipeline.py          # Flujo completo con reintentos y observabilidad
│   └── vector_store.py      # Creador y recuperador de Chroma VectorStore
├── tests/
│   └── test_pipeline.py     # Pruebas integrales automatizadas
├── .env.template            # Plantilla de variables de entorno
├── requirements.txt         # Librerías necesarias
└── README.md                # Esta guía
```

---

## Configuración y Setup

### 1. Prerrequisitos
- Python 3.10 o superior (el sistema corre bajo Python 3.14.6 en este entorno).
- Acceso a Internet para comunicarse con las APIs de OpenAI y Langfuse.

### 2. Variables de Entorno
Crea un archivo `.env` en el directorio raíz del proyecto basándote en la plantilla `.env.template`:

```env
OPENAI_API_KEY=tu_clave_de_openai
LANGFUSE_PUBLIC_KEY=tu_clave_publica_de_langfuse
LANGFUSE_SECRET_KEY=tu_clave_secreta_de_langfuse
LANGFUSE_HOST=https://cloud.langfuse.com
```

> [!NOTE]
> Puedes crear una cuenta gratuita en [Langfuse](https://cloud.langfuse.com) para obtener las claves de observabilidad y ver tu panel de control. Si no configuras las claves de Langfuse, el sistema funcionará igual de forma local desactivando el seguimiento remoto.

### 3. Instalación de Dependencias
Instala los paquetes necesarios usando `pip`:

```bash
pip install -r requirements.txt
```

---

## Ejecución del Sistema

### 1. Consola Interactiva
Para iniciar la interfaz interactiva de soporte:

```bash
python src/main.py
```

Al iniciarse, el sistema indexará automáticamente los archivos markdown ubicados en `company_docs/` en almacenes vectoriales locales dentro de `data/chroma/`. Después podrás ingresar consultas en la consola. Ejemplos de prueba:
- *¿Cuántos días de vacaciones al año tengo por ley en la empresa?* (HR)
- *¿Cuáles son los pasos para conectar la VPN Cisco AnyConnect?* (IT Support)
- *¿Cuánto me pueden reembolsar por comida de almuerzo en mis viajes de trabajo?* (Finance)
- *¿Bajo qué circunstancias se debe firmar un acuerdo de confidencialidad mutuo?* (Legal)

### 2. Ejecutar Pruebas Automatizadas
Puedes ejecutar las pruebas de integración para validar el ruteo inteligente y el grounding del RAG con el comando:

```bash
python -m unittest tests/test_pipeline.py
```
