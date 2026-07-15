import { create } from "zustand"

export interface DemoAttachment {
  file_id: string
  filename: string
  size_bytes: number
  uploaded_at: string
  /** Not present in the real AttachmentInfo -- kept in memory only, never sent anywhere but Gemini. */
  content: string
}

interface DemoAttachmentsState {
  filesByConversation: Record<string, DemoAttachment[]>
  addFile: (conversationId: string, file: DemoAttachment) => void
  removeFile: (conversationId: string, fileId: string) => void
  /** Mirrors the real backend consuming attachments after a successful send. */
  clearFiles: (conversationId: string) => void
}

/**
 * Demo-mode counterpart to the backend's ephemeral attachment storage
 * (src/context/attachments.py) -- see lib/demo.ts docstring for why this
 * exists. Deliberately NOT persisted to localStorage: these are meant to
 * live only as long as the tab, same lifetime as the real ones (consumed
 * on send, gone if the backend restarts).
 */
export const useDemoAttachmentsStore = create<DemoAttachmentsState>()((set) => ({
  filesByConversation: {},

  addFile: (conversationId, file) =>
    set((s) => ({
      filesByConversation: {
        ...s.filesByConversation,
        [conversationId]: [...(s.filesByConversation[conversationId] ?? []), file],
      },
    })),

  removeFile: (conversationId, fileId) =>
    set((s) => ({
      filesByConversation: {
        ...s.filesByConversation,
        [conversationId]: (s.filesByConversation[conversationId] ?? []).filter(
          (f) => f.file_id !== fileId,
        ),
      },
    })),

  clearFiles: (conversationId) =>
    set((s) => ({
      filesByConversation: { ...s.filesByConversation, [conversationId]: [] },
    })),
}))

// Referencia estable para "sin archivos". Un selector que devuelve `[]`
// (literal nuevo) cuando no hay entrada todavía crea un array DISTINTO
// en cada render; con la comparación por referencia de Zustand
// (Object.is), eso se lee como "cambió" en cada ciclo -> setState ->
// re-render -> el selector corre de nuevo -> nuevo array -> loop
// infinito ("Maximum update depth exceeded"). Devolver siempre la MISMA
// referencia cuando no hay archivos rompe el loop.
const EMPTY_FILES: DemoAttachment[] = []

/** Selector estable para usar con useDemoAttachmentsStore(selectDemoFiles(conversationId)). */
export function selectDemoFiles(conversationId: string | null) {
  return (s: DemoAttachmentsState): DemoAttachment[] =>
    conversationId ? (s.filesByConversation[conversationId] ?? EMPTY_FILES) : EMPTY_FILES
}
