import type { ChatMode } from "./api"

export type { ChatMode }

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
}

export interface GenerationOptionsState {
  temperature: number | null
  maxTokens: number | null
  thinkMode: boolean | null
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
  generation: GenerationOptionsState
}
