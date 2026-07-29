export interface RAGState {
  question: string
  mode: string
  collections: string[]
  results: RetrieveChunk[]
  confidence: number
  reformulated: boolean
  answer: string
  review_passed: boolean
  review_feedback: string
  review_attempts: number
  max_tokens: number | null
  think_mode: boolean | null
  extra: Record<string, unknown> | null
}

export type RAGStateUpdate = Partial<RAGState>

export interface RetrieveChunk {
  text: string
  source: string
  collection: string
  page: number
  score: number
}
