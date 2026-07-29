// Espejo de src/api/schemas/*.py (backend). Mantener sincronizado a mano
// -- son dos lenguajes distintos, no hay generación automática todavía.

export interface GenerationOptions {
  /**
   * La temperatura NO vive acá: es una propiedad fija de cada modelo,
   * definida en src/config/models/<backend>.py -- no hay override
   * por-request ni por-UI para eso (ver docstring del backend).
   *
   * Tampoco viven acá max_turns/top_k_initial/top_k_final: pasaron a
   * ser exclusivamente configuración de servidor (.env) -- ver
   * GenerationOptions en src/api/schemas/chat.py.
   */
  max_tokens?: number | null
  think_mode?: boolean | null
  /** Passthrough genérico sin validar en el cliente -- ver GenerationOptions.extra en el backend. */
  extra?: Record<string, unknown> | null
}

export type ChatMode = "SOFT" | "HARD"

export interface QueryRequest {
  question: string
  collections: string[]
  mode: ChatMode
  chat_history: { user: string; assistant: string }[]
  conversation_id?: string | null
  generation?: GenerationOptions | null
  /** Ver QueryRequest.web_search en el backend: complementa o reemplaza el contexto local. */
  web_search?: boolean
}

export interface WebSource {
  title: string
  url: string
}

export interface QueryResponse {
  answer: string
  confidence: number
  collections_used: string[]
  reformulated: boolean
  /** Lo que REALMENTE pasó, no lo que se pidió -- ver QueryResponse.used_web_search en el backend. */
  used_web_search: boolean
  web_sources: WebSource[] | null
  /** Se agotó la cuota de la cuenta de Tavily -- ver QueryResponse.web_search_quota_exceeded. */
  web_search_quota_exceeded: boolean
}

export interface CollectionsResponse {
  collections: string[]
}

export interface ProviderInfo {
  name: string
  model: string
  supports: string[]
  /**
   * Valor de thinking ya escrito en _MODELS[model] para este modelo --
   * null si el modelo no tiene modo de razonamiento (en ese caso
   * "think_mode" tampoco aparece en `supports`). El botón "Pensar" de
   * ResponseModeSection arranca reflejando esto, no un false fijo.
   */
  default_think: boolean | null
}

export interface ProvidersResponse {
  providers: Record<string, ProviderInfo>
  active_generation_provider: string
}

export interface EphemeralFileInfo {
  file_id: string
  filename: string
  chunk_count: number
  uploaded_at: string
}

export interface FileUploadResponse {
  conversation_id: string
  file_id: string
  filename: string
  chunk_count: number
  attached_to_collection?: string | null
}

export interface EphemeralFilesResponse {
  conversation_id: string
  files: EphemeralFileInfo[]
}

export interface DeleteResponse {
  deleted: boolean
}

// ---------------------------------------------------------------
// Archivos adjuntos ad-hoc -- espejo de src/api/schemas/attachments.py
// (AttachmentInfo se re-exporta ahí desde src/context/attachments.py,
// mismo patrón que EphemeralFileInfo/src/context/ephemeral.py).
// ---------------------------------------------------------------

export interface AttachmentInfo {
  file_id: string
  filename: string
  size_bytes: number
  uploaded_at: string
}

export interface AttachmentsResponse {
  files: AttachmentInfo[]
}

/**
 * Forma del `detail` de un 413 -- ver ContextLimitExceeded.as_detail()
 * en src/llm/context_guard.py. Devuelto tanto por /query, /query/agent
 * como por /query cuando cae en el caso sin colecciones/efímeros/web_search (mismo guard reutilizado en los tres casos, ver check_context_fit en src/llm/context_guard.py).
 */
/** Espejo de src/api/schemas/demo.py -- enviado a POST /api/v1/demo/query. */
export interface DemoQueryRequest {
  question: string
  collections: string[]
  mode: "SOFT" | "HARD"
}

export interface ContextLimitExceededDetail {
  error: "context_limit_exceeded"
  estimated_tokens: number
  limit: number
  model: string
}
