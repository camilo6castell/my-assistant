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
