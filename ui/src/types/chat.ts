import type { ChatMode, WebSource } from "./api"

export type { ChatMode, WebSource }

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  createdAt: number
  /** Solo en mensajes assistant exitosos. */
  confidence?: number
  collectionsUsed?: string[]
  reformulated?: boolean
  /** True si el mensaje assistant en realidad muestra un error (4xx/5xx). */
  isError?: boolean
  /** True mientras se espera la respuesta -- placeholder de "pensando...". */
  isPending?: boolean
  /** Texto junto a los puntos de carga mientras isPending (ej. "Revisando la web..."). */
  pendingLabel?: string
  /** Lo que REALMENTE pasó en este mensaje -- ver QueryResponse.used_web_search. */
  usedWebSearch?: boolean
  webSources?: WebSource[]
}

export interface GenerationOptionsState {
  temperature: number | null
  maxTokens: number | null
  thinkMode: boolean | null
  /** Overrides solo para esta conversación -- null = usar el default de .env/settings. */
  maxTurns: number | null
  topKInitial: number | null
  topKFinal: number | null
}

export interface Conversation {
  id: string
  title: string
  createdAt: number
  messages: ChatMessage[]
  activeCollections: string[]
  mode: ChatMode
  /** Usar /query/agent (grafo con revisión) en vez de /query (lineal). */
  useAgent: boolean
  /** Complementar (o reemplazar, sin colecciones) la respuesta con búsqueda web -- ver GenerationSection.tsx. */
  useWebSearch: boolean
  generation: GenerationOptionsState
}
