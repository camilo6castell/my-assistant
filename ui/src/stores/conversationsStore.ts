import { nanoid } from "nanoid"
import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { ChatMessage, ChatMode, Conversation } from "@/types/chat"

const emptyGeneration = {
  temperature: null,
  maxTokens: null,
  thinkMode: null,
  maxTurns: null,
  topKInitial: null,
  topKFinal: null,
}

function makeConversation(): Conversation {
  return {
    id: nanoid(),
    title: "Nueva conversación",
    createdAt: Date.now(),
    messages: [],
    activeCollections: [],
    mode: "SOFT",
    useAgent: false,
    generation: { ...emptyGeneration },
  }
}

/** ChatGPT/Claude-style: el título sale del primer mensaje del usuario. */
function titleFromMessage(content: string): string {
  const trimmed = content.trim().replace(/\s+/g, " ")
  return trimmed.length > 48 ? `${trimmed.slice(0, 48)}…` : trimmed || "Nueva conversación"
}

interface ConversationsState {
  conversations: Conversation[]
  activeId: string | null

  createConversation: () => string
  deleteConversation: (id: string) => void
  renameConversation: (id: string, title: string) => void
  setActive: (id: string) => void

  setActiveCollections: (id: string, collections: string[]) => void
  setMode: (id: string, mode: ChatMode) => void
  setUseAgent: (id: string, useAgent: boolean) => void
  setGeneration: (id: string, patch: Partial<Conversation["generation"]>) => void

  addMessage: (id: string, message: ChatMessage) => void
  updateMessage: (id: string, messageId: string, patch: Partial<ChatMessage>) => void
}

export const useConversationsStore = create<ConversationsState>()(
  persist(
    (set) => ({
      conversations: [],
      activeId: null,

      createConversation: () => {
        const conv = makeConversation()
        set((s) => ({ conversations: [conv, ...s.conversations], activeId: conv.id }))
        return conv.id
      },

      deleteConversation: (id) =>
        set((s) => {
          const remaining = s.conversations.filter((c) => c.id !== id)
          const activeId = s.activeId === id ? (remaining[0]?.id ?? null) : s.activeId
          return { conversations: remaining, activeId }
        }),

      renameConversation: (id, title) =>
        set((s) => {
          const trimmed = title.trim()
          if (!trimmed) return s // título vacío: no pisa el título existente
          return {
            conversations: s.conversations.map((c) =>
              c.id === id ? { ...c, title: trimmed } : c
            ),
          }
        }),

      setActive: (id) => set({ activeId: id }),

      setActiveCollections: (id, collections) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, activeCollections: collections } : c
          ),
        })),

      setMode: (id, mode) =>
        set((s) => ({
          conversations: s.conversations.map((c) => (c.id === id ? { ...c, mode } : c)),
        })),

      setUseAgent: (id, useAgent) =>
        set((s) => ({
          conversations: s.conversations.map((c) => (c.id === id ? { ...c, useAgent } : c)),
        })),

      setGeneration: (id, patch) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...c, generation: { ...c.generation, ...patch } } : c
          ),
        })),

      addMessage: (id, message) =>
        set((s) => ({
          conversations: s.conversations.map((c) => {
            if (c.id !== id) return c
            const isFirstUserMessage = c.messages.length === 0 && message.role === "user"
            return {
              ...c,
              messages: [...c.messages, message],
              title: isFirstUserMessage ? titleFromMessage(message.content) : c.title,
            }
          }),
        })),

      updateMessage: (id, messageId, patch) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === messageId ? { ...m, ...patch } : m
                  ),
                }
              : c
          ),
        })),
    }),
    { name: "myassistant-conversations" }
  )
)
