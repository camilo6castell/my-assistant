// Espejo de src/api/schemas/*.py (backend). Mantener sincronizado a mano
// -- son dos lenguajes distintos, no hay generación automática todavía.

export interface GenerationOptions {
  temperature?: number | null
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
}

export interface QueryResponse {
  answer: string
  confidence: number
  collections_used: string[]
  reformulated: boolean
}

export interface CollectionsResponse {
  collections: string[]
}

export interface ProviderInfo {
  name: string
  model: string
  supports: string[]
}

export interface GenerationDefaults {
  temperature: number
  max_turns: number
  hard_top_k_initial: number
  hard_top_k_final: number
  soft_top_k_initial: number
  soft_top_k_final: number
}

export interface ProvidersResponse {
  providers: Record<string, ProviderInfo>
  active_generation_provider: string
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
