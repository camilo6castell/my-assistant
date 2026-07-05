// Espejo de src/api/schemas/*.py (backend). Mantener sincronizado a mano
// -- son dos lenguajes distintos, no hay generación automática todavía.

export interface GenerationOptions {
  temperature?: number | null
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
