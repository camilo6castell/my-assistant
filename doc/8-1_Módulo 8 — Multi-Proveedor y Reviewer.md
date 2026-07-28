# Módulo 8 — Multi-Proveedor y Reviewer _(nuevo)_

Este módulo documenta dos piezas que surgieron del mismo problema: el sistema original solo podía usar un modelo LLM a la vez. Para que `reformulate_node` use Gemini y `generate_node` use el modelo local, hacía falta una arquitectura de proveedores, no solo cambiar una variable de entorno.

---

## El problema original

El `client.py` original era esto:

```python
client = OpenAI(
    base_url=settings.llm_base_url,   # un solo endpoint
    api_key=settings.llm_api_key,
)
```

Un cliente global. Para usar Gemini, tenías que editar `.env` y reiniciar el proceso. No era posible tener dos modelos activos simultáneamente en el mismo grafo.

El sistema también tenía `.env.gemini` con credenciales de Gemini en texto plano, y `.gitignore` solo excluía `.env` — no `.env.gemini` ni `.env.local`. Eso es un riesgo de seguridad real: un `git push` accidental hubiera expuesto la API key.

---

## `providers.py` — el registro de proveedores

```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
```

Cada proveedor es solo estas cuatro cosas. La configuración viene de `settings`, que lee `.env.providers`.

```python
def _build_provider_table() -> dict[str, ProviderConfig]:
    return {
        "local": ProviderConfig(
            name="local",
            base_url=settings.local_base_url,
            api_key=settings.local_api_key,
            model=settings.local_model,
        ),
        "gemini": ProviderConfig(
            name="gemini",
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        ),
        # Para agregar un tercer proveedor:
        # "claude": ProviderConfig(
        #     name="claude",
        #     base_url=settings.claude_base_url,
        #     api_key=settings.claude_api_key,
        #     model=settings.claude_model,
        # ),
    }
```

La tabla se construye en una función (no a nivel de módulo) para que el test que mute `settings` vea los valores actualizados, no los del momento del import.

**El cliente cacheado:**

```python
@lru_cache(maxsize=None)
def _client_for(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)
```

`@lru_cache` garantiza que se crea un único `OpenAI` client por `(base_url, api_key)` único — no uno por request, sino uno por proveedor para toda la vida del proceso. Crear un cliente HTTP tiene overhead (pool de conexiones, negociación TLS). Cachearlo es correcto.

**La interfaz pública:**

```python
def get_client(name: str) -> tuple[OpenAI, ProviderConfig]:
    config = get_provider(name)
    client = _client_for(config.base_url, config.api_key)
    return client, config
```

Los nodos del grafo no importan `providers.py` directamente — acceden a él a través de `ask_llm()` y `ask_llm_internal()`. Eso mantiene a `nodes.py` desacoplado de los detalles de la librería OpenAI.

---

## Routing de proveedores por nodo

La decisión de qué proveedor usa cada nodo vive en `.env.providers`:

```env
REFORMULATE_PROVIDER=gemini
GENERATE_PROVIDER=local
```

Y en el código, cada nodo lo pasa explícitamente:

```python
# reformulate_node
ask_llm_internal(prompt=..., provider=settings.reformulate_provider)

# generate_node
ask_llm(prompt=..., chat_memory=..., provider=settings.generate_provider)

# review_node (mismo proveedor que reformulate: Gemini)
ask_llm_internal(prompt=review_prompt, provider=settings.reformulate_provider)

# correct_node (mismo proveedor que generate: local)
ask_llm(prompt=correction_prompt, chat_memory=..., provider=settings.generate_provider)
```

El proveedor se pasa explícitamente en cada call para que el código sea legible sin necesidad de trazar la configuración: al leer `nodes.py` queda claro qué nodo usa qué modelo.

**Por qué el modelo local siempre genera la respuesta final:**

Los chunks de tus documentos indexados viajan en el prompt de `generate_node`. Si ese nodo usara Gemini, esos chunks — potencialmente privados (notas personales, libros, documentación interna) — saldrían de tu máquina. El modelo local garantiza que el contenido de tus colecciones nunca se envía a ningún proveedor externo.

Gemini solo recibe:

- En `reformulate_node`: la pregunta del usuario (texto corto, sin contexto de documentos).
- En `review_node`: la respuesta generada + los chunks (texto ya generado, no documentos crudos).

Puedes ajustar este comportamiento para colecciones que no sean sensibles simplemente cambiando `GENERATE_PROVIDER=gemini` en `.env.providers` — sin tocar código.

---

## El ciclo review → correct

Este es el mecanismo de auto-corrección. El objetivo es reducir alucinaciones y forzar citación de fuentes sin que el usuario tenga que hacer seguimiento manual.

**Qué evalúa `review_node`:**

1. **Anclaje:** ¿cada afirmación de la respuesta aparece en los chunks recuperados? Si el modelo dice algo que no está en el contexto, es una alucinación.
2. **Citas:** ¿la respuesta menciona de qué fuente viene la información? Un RAG que no cita no es útil para trabajo académico o de investigación.

**El protocolo JSON:**

```json
{"passed": true}
{"passed": false, "feedback": "La respuesta afirma que Freud publicó 1984, pero eso no aparece en el contexto."}
```

Gemini recibe instrucción de responder solo con JSON, sin texto adicional. Pero Gemini a veces envuelve la respuesta en bloques de código Markdown (` ```json ... ``` `) aunque se le pida que no. `review_node` tiene una limpieza defensiva antes de parsear:

````python
clean = raw.strip() \
    .removeprefix("```json") \
    .removeprefix("```") \
    .removesuffix("```") \
    .strip()
result = json.loads(clean)
````

**Fallo silencioso explícito:**

Si Gemini devuelve un 503 (alta demanda del modelo), `ask_llm_internal` captura la excepción y devuelve el prompt completo como string (el fallback). Ese string no es JSON válido. `review_node` detecta el error de parsing y aprueba por defecto — el usuario recibe la respuesta sin revisar en vez de no recibir nada.

Este comportamiento es intencional y está documentado en el código. El log deja evidencia:

```
[WARNING] [graph] review_node | no se pudo parsear JSON → aprobando por defecto
```

**`MAX_REVIEW_ATTEMPTS`:**

```python
MAX_REVIEW_ATTEMPTS = 1   # en nodes.py
```

Si el reviewer rechaza y `correct_node` corrige, el ciclo vuelve a `review_node`. Si esta segunda revisión también falla, el cap forza `review_passed=True` — el loop se rompe y la respuesta llega al usuario. Esto evita que un reviewer excesivamente estricto o un modelo local inconsistente bloqueen al usuario indefinidamente.

El valor `1` significa: máximo un reintento. Si quieres dos correcciones posibles, sube a `2`. El costo es más latencia y más tokens.

---

## Gestión de secretos: qué cambió y por qué importa

El `.env.gemini` original tenía la API key en texto plano y no estaba excluido del repositorio. Se corrigió así:

**`.gitignore` ahora excluye todo `.env.*`:**

```gitignore
.env
.env.*
!.env.example
```

La excepción `!.env.example` permite que el template sin credenciales sí esté en el repo.

**`.env.providers` reemplaza a `.env.gemini`:**

Un archivo dedicado a la configuración multi-proveedor, con un placeholder `REPLACE_ME_ROTATE_THIS_KEY` en lugar de la key real.

**`.env.example` sin secretos:**

Template que se puede commitear y usar como referencia de onboarding. No tiene ningún valor real — solo nombres de variables y comentarios.

Si en algún momento commiteas accidentalmente un `.env.*` con credenciales reales, el daño mitigation es:

1. Rotar la key inmediatamente en el dashboard del proveedor.
2. Borrar el commit del historial de git (`git filter-branch` o `git rebase -i`).
3. Verificar que el nuevo `.gitignore` previene que vuelva a ocurrir.

---

## Cómo agregar un tercer proveedor

Supongamos que quieres agregar Claude de Anthropic como opción:

**1. En `providers.py`:**

```python
"claude": ProviderConfig(
    name="claude",
    base_url=settings.claude_base_url,
    api_key=settings.claude_api_key,
    model=settings.claude_model,
),
```

**2. En `settings.py`:**

```python
claude_base_url: str = Field(default="https://api.anthropic.com/v1")
claude_api_key: str = Field(default="")
claude_model: str = Field(default="claude-opus-4-6")
```

**3. En `.env.providers`:**

```env
CLAUDE_BASE_URL=https://api.anthropic.com/v1
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-6
```

**4. Para usarlo en un nodo:**

```env
REFORMULATE_PROVIDER=claude
```

Cero cambios en `graph.py`, `nodes.py`, ni `generate.py`. El proveedor nuevo es una entrada en una tabla y tres variables de entorno.

---

## Talking points para entrevista

**"¿Cómo manejas múltiples proveedores LLM?"**

> Patrón provider-per-node: factory con caché de clientes, configuración por variables de entorno, cero cambios en la lógica del grafo al agregar un proveedor. El proveedor del nodo se pasa explícitamente para que sea legible sin trazar configuración.

**"¿Cómo evitas que los documentos privados salgan de tu máquina?"**

> El proveedor que genera la respuesta final siempre es local. Los proveedores externos solo reciben la pregunta del usuario (reformulación) o el texto de la respuesta ya generada (review) — nunca los chunks de los documentos indexados.

**"¿Cómo manejas la posibilidad de que el reviewer falle?"**

> Fallo silencioso explícito: si el proveedor externo devuelve un error o una respuesta no parseable, se aprueba por defecto. El log registra el evento. Un loop cap (`MAX_REVIEW_ATTEMPTS`) evita loops infinitos independientemente del resultado.

**"¿Cómo manejas secretos en este proyecto?"**

> `.gitignore` excluye todo `.env.*` salvo `.env.example`. Los archivos con credenciales reales nunca entran al repositorio. Hay un template sin valores reales para onboarding. Si una key se expone accidentalmente, el primer paso es rotarla en el dashboard del proveedor.
