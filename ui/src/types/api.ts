// Espejo de src/api/schemas/*.py (backend). Mantener sincronizado a mano
// -- son dos lenguajes distintos, no hay generación automática todavía.

export interface GenerationOptions {
  /**
   * La temperatura NO vive acá: es una propiedad fija de cada modelo,
   * definida en src/config/models/<backend>.py -- no hay override
   * por-request ni por-UI para eso (ver docstring del backend).
   */
  max_tokens?: number | null
  think_mode?: boolean | null
  /** Passthrough genérico sin validar en el cliente -- ver GenerationOptions.extra en el backend. */
  extra?: Record<string, unknown> | null
  /** Ventana de historial (turnos) enviada al LLM. null = settings.max_turns. */
  max_turns?: number | null
  /** Overrides de retrieval, dependientes del modo SOFT/HARD del request. */
  top_k_initial?: number | null
  top_k_final?: number | null
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
   * GenerationSection arranca reflejando esto, no un false fijo.
   */
  default_think: boolean | null
}

export interface GenerationDefaults {
  max_turns: number
  hard_top_k_initial: number
  hard_top_k_final: number
  soft_top_k_initial: number
  soft_top_k_final: number
}

export interface ProvidersResponse {
  providers: Record<string, ProviderInfo>
  active_generation_provider: string
  /** Mapa completo rol -> provider (ver LLMRole en el backend). */
  provider_roles: Record<string, string>
  defaults: GenerationDefaults
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
