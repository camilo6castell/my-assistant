# Módulo 5 — Generación y Prompts

Tenemos 7 `SearchResult` ordenados por relevancia, con texto, fuente y página. El trabajo de este módulo es convertir eso en un único string de texto — el prompt — que se envía al LLM, y luego gestionar esa llamada.

Hay un punto conceptual importante antes de entrar al código: **el LLM no "ve" tus documentos**. No tiene acceso a `vector_stores/`, no sabe qué es FAISS, no tiene memoria de conversaciones anteriores. Todo lo que el LLM conoce en el momento de responder es exactamente el texto que está dentro del string `prompt`. Este módulo es, literalmente, la construcción de la realidad completa del LLM para esa consulta.

---

## De `SearchResult` a `context_chunks`

Este paso ocurre en `interface.py`, justo antes de llamar a `build_prompt`:

```python
context_chunks: list[str] = [
    f"FUENTE: {r.source}\nCOLECCION: {r.collection}\nPAGINA: {r.page}\n\n{r.text}"
    for r in results
]
```

Cada `SearchResult` (un objeto estructurado con campos) se aplana a un string con formato fijo:

```
FUENTE: La-sociedad-del-espectaculo.pdf
COLECCION: sociologia/debord
PAGINA: 47

El espectáculo no es un conjunto de imágenes, sino una
relación social entre personas mediatizada por imágenes...
```

¿Por qué incluir `FUENTE`, `COLECCION` y `PAGINA` como texto plano dentro del contenido? Porque es la única forma de que el LLM "sepa" de dónde viene cada fragmento. Si tu prompt final menciona estas etiquetas y le pides al LLM que cite fuentes, el LLM puede literalmente copiar esos valores en su respuesta — no porque entienda tu sistema de archivos, sino porque están ahí, en el texto que está leyendo.

---

## `build_prompt()` — el ensamblador final

```python
def build_prompt(
    context_chunks: list[str],
    question: str,
    mode: str,
    chat_memory: list[TurnMemory],
) -> str:
    return f"""
Eres un asistente RAG.

Tu tarea es responder preguntas usando
EXCLUSIVAMENTE el contexto proporcionado.

{build_rules_block(mode)}

========================================
HISTORIAL
========================================

{build_history_block(chat_memory)}

========================================
CONTEXTO
========================================

{build_context_block(context_chunks)}

========================================
PREGUNTA
========================================

{question}

========================================
RESPUESTA
========================================
"""
```

Esta función no tiene lógica compleja — es un f-string que concatena 4 secciones. Pero el **orden y la estructura** son decisiones de diseño deliberadas. Vamos sección por sección.

---

### Sección 1 — `build_rules_block(mode)`: las reglas cambian el comportamiento

```python
def build_rules_block(mode: str) -> str:
    if mode == ChatMode.INTERPRETATIVE:
        return """
REGLAS (MODO INTERPRETATIVO):

- Puedes conectar ideas entre múltiples fuentes.
- Puedes sintetizar conceptos.
- Puedes abstraer principios generales.
- Puedes explicar implicaciones teóricas.
- Mantente fiel al contexto.
- Nunca inventes información externa.
- Indica fuentes cuando sea posible.
"""

    return """
REGLAS (MODO RIGUROSO):

- Usa únicamente el contenido presente en el contexto.
- No inventes información.
- No uses conocimiento externo.
- Si algo no está en el contexto, dilo explícitamente.
- Prioriza precisión textual.
"""
```

Este es un punto clave para tu entrevista: **el "modo" del sistema no es un parámetro técnico del modelo (no cambia la temperatura, ni el modelo, ni nada de la API)**. Es exclusivamente texto en el prompt que le indica al LLM cómo comportarse.

Recuerda que en el Módulo 4 vimos que el modo INTERPRETATIVO también afecta cuántos chunks se recuperan (`top_k`) y cuántas variantes de query se generan. Así que "modo" tiene dos efectos:

```
MODO RIGUROSO       → menos chunks recuperados + instrucción "cíñete al texto"
MODO INTERPRETATIVO → más chunks recuperados + instrucción "puedes sintetizar"
```

Ambos efectos refuerzan la misma intención desde ángulos distintos: uno controla *qué información llega*, el otro controla *qué se le permite hacer con ella*.

---

### Sección 2 — `build_history_block()`: la memoria conversacional

```python
def build_history_block(chat_memory: list[TurnMemory]) -> str:
    if not chat_memory:
        return "No hay historial previo."

    lines: list[str] = []

    for turn in chat_memory[-MAX_TURNS:]:
        lines.append(f"Usuario: {turn['user']}")
        lines.append(f"Asistente: {turn['assistant']}")
        lines.append("")

    return "\n".join(lines)
```

Aquí aparece algo que ya conoces de Java/JS pero vale la pena nombrarlo explícitamente: **el LLM no tiene estado entre llamadas**. Cada llamada a la API es completamente independiente — es una función pura, sin memoria. Si quieres que el modelo "recuerde" lo que dijiste hace dos mensajes, tienes que **reenviar esa conversación completa cada vez**.

`chat_memory` es una lista de turnos que vive en `ChatSession` (la veremos en el Módulo 6). Cada turno es:

```python
TurnMemory(user="¿qué es el espectáculo?", assistant="Según Debord, el espectáculo es...")
```

`chat_memory[-MAX_TURNS:]` — slicing para tomar solo los últimos `MAX_TURNS` (4 por defecto) turnos. ¿Por qué no enviar todo el historial?

**Razón:** los LLMs tienen una ventana de contexto limitada (medida en tokens). Cada turno de historial consume tokens que ya no están disponibles para el contexto recuperado o la respuesta. Con `MAX_TURNS=4`, garantizas que el historial nunca crece sin límite — es una ventana deslizante, como un buffer circular de tamaño fijo.

```
Turno 1, 2, 3, 4, 5, 6, 7...
                  └────┴────┴────┴────┘
                   solo estos 4 van al prompt
```

---

### Sección 3 — `build_context_block()`: el contenido recuperado

```python
def build_context_block(context_chunks: list[str]) -> str:
    return "\n\n---\n\n".join(context_chunks)
```

La función más simple del módulo, pero merece atención: simplemente une los 7 chunks (cada uno con su `FUENTE/COLECCION/PAGINA/texto`) separados por `---`. El separador visual ayuda al LLM a distinguir dónde termina un fragmento y empieza el siguiente — sin esto, el modelo podría "fusionar" mentalmente dos fragmentos de fuentes distintas como si fueran continuos.

---

### Sección 4 — La pregunta y el cierre

```python
========================================
PREGUNTA
========================================

{question}

========================================
RESPUESTA
========================================

```

El prompt termina literalmente con la palabra `RESPUESTA` seguida de un salto de línea — y nada más. Esto es una técnica deliberada: le da al modelo una señal visual clara de "aquí termina la entrada, aquí empieza tu salida". Es similar a cómo en un patrón de prompt engineering se usa `Answer:` al final para anclar el formato de respuesta.

---

## El prompt completo ensamblado — visualización

```
Eres un asistente RAG.
Tu tarea es responder preguntas usando EXCLUSIVAMENTE el contexto proporcionado.

REGLAS (MODO INTERPRETATIVO):
- Puedes conectar ideas entre múltiples fuentes...

========================================
HISTORIAL
========================================
Usuario: ¿qué es el espectáculo según Debord?
Asistente: Según Debord, el espectáculo es la forma actual...

========================================
CONTEXTO
========================================
FUENTE: La-sociedad-del-espectaculo.pdf
COLECCION: sociologia/debord
PAGINA: 47

El espectáculo no es un conjunto de imágenes...

---

FUENTE: La-interpretacion-de-los-suenos.pdf
COLECCION: psicoanalisis/freud
PAGINA: 112

El inconsciente se manifiesta a través de...

---

[... 5 chunks más ...]

========================================
PREGUNTA
========================================
¿Cómo se relacionan el espectáculo y el inconsciente?

========================================
RESPUESTA
========================================
```

Este string completo — potencialmente varios miles de caracteres — es **una sola string de Python** que viaja como un solo "mensaje" hacia el LLM.

---

## `generate.py` — la llamada al modelo

Ahora que tenemos el `prompt`, ¿cómo llega al LLM?

```python
SYSTEM_PROMPT: str = """
Eres un asistente RAG especializado en responder
usando únicamente el contexto proporcionado.

Reglas:
- Prioriza el contenido recuperado.
- No inventes información.
...
"""

def build_messages(
    prompt: str,
    chat_memory: list[TurnMemory],
) -> list[ChatCompletionMessageParam]:

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for turn in chat_memory:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": prompt})

    return messages
```

Aquí hay algo que vale la pena notar: **el historial se incluye dos veces, en dos formatos distintos**.

```
build_messages():
  ├─ chat_memory → mensajes separados con role "user"/"assistant"  (formato API)
  └─ prompt      → ya contiene HISTORIAL embebido como texto       (formato texto)
```

Esto es en realidad redundante — el modelo recibe el historial tanto en la estructura nativa de la API de chat (`messages: [...]`) como dentro del texto del prompt mismo. No es un bug crítico, pero es algo que podrías identificar como mejora si te preguntan "¿qué optimizarías?": eliminar la redundancia eligiendo un solo mecanismo.

**El array `messages` resultante:**

```python
[
    {"role": "system",    "content": "Eres un asistente RAG..."},
    {"role": "user",      "content": "¿qué es el espectáculo?"},      # turno 1
    {"role": "assistant", "content": "Según Debord..."},               # turno 1
    {"role": "user",      "content": "<<< el prompt gigante con CONTEXTO+PREGUNTA >>>"},
]
```

Esta estructura `[{role, content}, ...]` es el formato estándar de la API de Chat Completions — el mismo que usan OpenAI, Ollama, y tu FastFlowLM local. Es universal porque todos estos sistemas implementan la misma interfaz HTTP.

---

## La llamada HTTP real

```python
def ask_llm(prompt: str, chat_memory: list[TurnMemory]) -> str:

    logger.info(
        f"Consultando LLM | model={LLM_MODEL} "
        f"| timeout={LLM_TIMEOUT}s | temp={LLM_TEMPERATURE}"
    )

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=build_messages(prompt=prompt, chat_memory=chat_memory),
            temperature=LLM_TEMPERATURE,
            timeout=LLM_TIMEOUT,
        )

        content = response.choices[0].message.content

        if not content:
            return "El modelo no devolvió respuesta."

        return str(content).strip()

    except OpenAIError as e:
        logger.exception("Error consultando LLM")
        return f"Error consultando modelo: {e}"
```

`client` viene de `src/llm/client.py`:

```python
client = OpenAI(
    base_url=LLM_BASE_URL,   # http://127.0.0.1:52625/v1  ← tu FastFlowLM local
    api_key=LLM_API_KEY,     # "flm" — cualquier string, FastFlowLM no valida
)
```

Aunque uses un modelo corriendo localmente en tu NPU vía FastFlowLM, el cliente Python es el SDK oficial de **OpenAI** — porque FastFlowLM (igual que Ollama) implementa el mismo contrato HTTP que la API de OpenAI (`/v1/chat/completions`). El SDK no sabe ni le importa que el servidor al otro lado sea un modelo corriendo en tu Ryzen AI 365 en lugar de los servidores de OpenAI. Esto es lo que se llama una **API OpenAI-compatible** — un estándar de facto en el ecosistema de LLMs locales.

**`temperature`** controla cuánta aleatoriedad hay en la elección de palabras del modelo — valores bajos (0.2 aquí) hacen respuestas más deterministas y conservadoras, lo cual tiene sentido para un sistema RAG donde quieres precisión sobre creatividad.

**`timeout`** — recordarás de una sesión anterior que esto estaba en 120s y lo subimos a 600s, porque un modelo cuantizado local con un prompt de varios miles de caracteres (con 7 chunks de contexto) puede tardar varios minutos.

---

## El flujo completo del Módulo 5

```
7 SearchResult
     │
     ▼  format por resultado (FUENTE/COLECCION/PAGINA/texto)
context_chunks: list[str]
     │
     ▼  build_prompt()
     │     ├─ build_rules_block(mode)        → reglas según RIGUROSO/INTERPRETATIVO
     │     ├─ build_history_block(memory)    → últimos 4 turnos
     │     ├─ build_context_block(chunks)    → 7 chunks unidos con "---"
     │     └─ question
     │
     ▼
prompt: str  (un solo string gigante)
     │
     ▼  build_messages()
     │     ├─ system prompt
     │     ├─ historial como mensajes user/assistant
     │     └─ prompt como último mensaje "user"
     │
     ▼
messages: list[dict]
     │
     ▼  client.chat.completions.create(model, messages, temperature, timeout)
     │     (HTTP POST → http://127.0.0.1:52625/v1/chat/completions)
     │
     ▼
response.choices[0].message.content
     │
     ▼
answer: str  ← esto es lo que ve el usuario
```

---

¿Avanzamos al Módulo 6 — el último: cómo `ChatSession` mantiene todo este estado cohesionado y cómo el loop de `interface.py` conecta cada pieza que ya vimos?
