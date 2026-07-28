# Módulo 5b — Validación de Contexto y Tokens

Este módulo implementa un guardia preventivo que verifica si el request completo (system prompt + contexto + historial + pregunta) cabe en la ventana de contexto del modelo **antes** de hacer la llamada al LLM. Si no cabe, rechaza el request con un error claro en lugar de esperar a que el modelo recorte la respuesta a mitad de camino.

---

## Estimación de tokens — `src/utils/tokens.py`

No existe un tokenizer exacto para todos los backends (Qwen via FastFlowLM, DeepSeek via Ollama, Gemini) sin traer tres bibliotecas diferentes. El módulo usa `tiktoken` (tokenizador OpenAI, codificación `cl100k_base`) como aproximación razonable para texto en inglés y español.

```python
import tiktoken

HEURISTIC_CHARS_PER_TOKEN = 3.5

_encoder: tiktoken.Encoding | None

try:
    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoder = None
    logger.warning(
        "[tokens] tiktoken could not load its encoding (no network to download "
        "host) -- using character heuristic to estimate tokens."
    )


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if _encoder is not None:
        return len(_encoder.encode(text, disallowed_special=()))
    return int(len(text) / HEURISTIC_CHARS_PER_TOKEN)
```

**¿Por qué `tiktoken` tiende a sobreestimar?** Los tokenizadores SentencePiece (Qwen/Gemini) suelen producir menos tokens por carácter. Para un guardia preventivo, sobreestimar es el sesgo correcto: mejor advertir de más que dejar pasar un request que el modelo rechazaría.

**Fallback sin red:** si `tiktoken` no puede cargar su archivo de codificación (falta conexión a la URL de descarga), usa una heurística de caracteres. El factor 3.5 (en lugar de 4) compensa que el español tiene más acentos y palabras largas que el inglés.

**¿Por qué no usar el tokenizer exacto del modelo?** Porque cada backend tiene el suyo propio y mantener tres dependencias no vale la pena para un guardia preventivo que solo necesita ser conservador.

---

## El context guard — `src/nlp/llm/context_guard.py`

### Constantes configurables

```python
SAFETY_MARGIN_RATIO = settings.context_guard_safety_margin
DEFAULT_OUTPUT_RESERVE = settings.context_guard_output_reserve
```

- **`SAFETY_MARGIN_RATIO`** (10% por defecto): margen de seguridad sobre la estimación de tokens. Absorbe la imprecisión del estimador.
- **`DEFAULT_OUTPUT_RESERVE`** (1024 tokens): tokens reservados para la respuesta cuando el request no especifica `max_tokens`. Un piso conservador para que el modelo nunca se quede sin espacio para responder.

### `ContextLimitExceeded` — la excepción

```python
class ContextLimitExceeded(Exception):
    def __init__(self, *, estimated_tokens: int, limit: int, model: str) -> None:
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        self.model = model
        super().__init__(
            f"Estimated request ({estimated_tokens} tokens) exceeds the usable "
            f"limit ({limit} tokens) of model '{model}'."
        )

    def as_detail(self) -> dict[str, int | str]:
        return {
            "error": "context_limit_exceeded",
            "estimated_tokens": self.estimated_tokens,
            "limit": self.limit,
            "model": self.model,
        }
```

Se levanta cuando el request estimado excede la ventana de contexto del modelo. El router la captura y la traduce a un HTTP 413 via `as_detail()`.

### `check_context_fit()` — la función principal

```python
def check_context_fit(
    *,
    system_prompt: str,
    prompt: str,
    chat_memory: list[TurnMemory],
    provider: str,
    max_tokens: int | None,
) -> None:
```

Flujo paso a paso:

1. **Resuelve el cliente y la configuración** del proveedor:
   ```python
   _, config = get_client(provider)
   limit = get_context_window(config.capabilities, config.model)
   ```

2. **Fail-open si no hay `context_window` configurado:**
   ```python
   if limit is None:
       return
   ```
   Si el modelo aún no tiene documentada su ventana de contexto en `src/config/models/<backend>.py`, la verificación pasa. Es preferible a bloquear requests por configuración incompleta. Llenar ese valor es lo que activa el guardia para ese modelo.

3. **Estima los tokens totales:**
   ```python
   text_parts = [system_prompt, prompt]
   for turn in chat_memory:
       text_parts.append(turn.user)
       text_parts.append(turn.assistant)

   estimated = sum(estimate_tokens(part) for part in text_parts)
   estimated = int(estimated * (1 + SAFETY_MARGIN_RATIO))
   ```
   Suma system prompt, prompt del usuario, y cada turno del historial. Aplica el margen de seguridad.

4. **Calcula el espacio usable:**
   ```python
   reserve = max_tokens or DEFAULT_OUTPUT_RESERVE
   usable = limit - reserve
   ```
   Resta del límite total el espacio reservado para la respuesta.

5. **Verifica y lanza si no cabe:**
   ```python
   if estimated > usable:
       raise ContextLimitExceeded(
           estimated_tokens=estimated,
           limit=usable,
           model=config.model,
       )
   ```

---

## Integración con el grafo

El guardia se ejecuta **antes** de cada llamada LLM en los nodos que generan texto.

### En `generate_node` (respuesta principal)

```python
# src/graph/nodes.py — generate_node
check_context_fit(
    system_prompt=build_system_prompt(),
    prompt=prompt,
    chat_memory=state["chat_memory"],
    provider=LLMRole.GENERATE.value,
    max_tokens=state.get("max_tokens"),
)

answer = ask_llm(
    prompt=prompt,
    chat_memory=state["chat_memory"],
    provider=LLMRole.GENERATE.value,
    max_tokens=state.get("max_tokens"),
    think_mode=state.get("think_mode"),
    extra=state.get("extra"),
)
```

El `check_context_fit()` corre después de construir el prompt (que incluye los chunks recuperados) pero antes de enviarlo al LLM. Esto es porque el prompt con el contexto recuperado solo existe en este punto — antes de invocar el grafo, el router no sabe qué se va a recuperar.

Si `ContextLimitExceeded` se levanta, **propaga como-is** hasta el router, que lo traduce a un HTTP 413 con el detalle del error.

### En el router HTTP (API)

```python
# src/api/routers/chat.py
try:
    check_context_fit(...)
except ContextLimitExceeded as e:
    raise HTTPException(status_code=413, detail=e.as_detail())
```

El router captura la excepción y la convierte en una respuesta HTTP con código 413 (Payload Too Large) y un body JSON descriptivo:

```json
{
  "error": "context_limit_exceeded",
  "estimated_tokens": 15200,
  "limit": 8192,
  "model": "qwen3.5:9b"
}
```

---

## Resumen: por qué este diseño

El context guard existe porque los modelos tienen ventanas de contexto finitas y un prompt de RAG puede crecer rápido (muchos chunks recuperados + historial extenso). Sin esta validación, el modelo recibiría un request demasiado grande y produciría una respuesta truncada a mitad de frase — un fallo silencioso y difícil de diagnosticar.

Al validar **antes** de la llamada LLM:

- El usuario recibe un error claro y accionable (reducir contexto, limpiar historial).
- El sistema no desperdicia tokens de LLM en un request que no va a caber.
- El router puede traducir el error a un HTTP estándar (413) que el frontend entiende.
- No hay truncamiento silencioso ni respuestas cortadas a mitad de camino.

El enfoque "fail-open" (si no hay `context_window` configurado, el guardia no bloquea nada) prioriza la disponibilidad sobre la exactitud: es mejor servir un request arriesgado que rechazar todos los requests de un modelo cuya configuración aún no está completa.
