import { useMutation } from "@tanstack/react-query"
import { nanoid } from "nanoid"
import { Navigate, useParams } from "react-router-dom"
import { useCollections } from "@/hooks/useCollections"
import { useEphemeralFiles } from "@/hooks/useEphemeralFiles"
import { useProviders } from "@/hooks/useProviders"
import { apiErrorMessage, postQuery } from "@/lib/api/client"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ChatMessage } from "@/types/chat"
import { ChatToolbar } from "./ChatToolbar"
import { MessageInput } from "./MessageInput"
import { MessageList } from "./MessageList"

function toHistory(messages: ChatMessage[]): { user: string; assistant: string }[] {
  const history: { user: string; assistant: string }[] = []
  for (let i = 0; i < messages.length - 1; i++) {
    const a = messages[i]
    const b = messages[i + 1]
    if (a.role === "user" && b.role === "assistant" && !b.isPending && !b.isError) {
      history.push({ user: a.content, assistant: b.content })
    }
  }
  return history
}

export function ChatView() {
  const { conversationId } = useParams<{ conversationId: string }>()
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId)
  )
  const addMessage = useConversationsStore((s) => s.addMessage)
  const updateMessage = useConversationsStore((s) => s.updateMessage)

  const { data: collectionsData } = useCollections()
  const { data: providersData } = useProviders()
  const ephemeralFiles = useEphemeralFiles(conversationId ?? null)

  const sendMutation = useMutation({
    mutationFn: (text: string) => {
      if (!conversation) throw new Error("Conversación no encontrada")
      return postQuery(
        {
          question: text,
          collections: conversation.activeCollections,
          mode: conversation.mode,
          chat_history: toHistory(conversation.messages),
          conversation_id: conversation.id,
          generation: {
            temperature: conversation.generation.temperature,
            max_tokens: conversation.generation.maxTokens,
            think_mode: conversation.generation.thinkMode,
          },
        },
        conversation.useAgent
      )
    },
  })

  // El hook de arriba no puede ser condicional -- por eso el early return
  // va después de declarar todos los hooks, no antes.
  if (!conversationId || !conversation) {
    return <Navigate to="/" replace />
  }

  function handleSend(text: string) {
    if (!conversation) return
    const userMsg: ChatMessage = {
      id: nanoid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    }
    const assistantId = nanoid()
    const pendingMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      isPending: true,
    }

    addMessage(conversation.id, userMsg)
    addMessage(conversation.id, pendingMsg)

    sendMutation.mutate(text, {
      onSuccess: (data) => {
        updateMessage(conversation.id, assistantId, {
          content: data.answer,
          confidence: data.confidence,
          collectionsUsed: data.collections_used,
          reformulated: data.reformulated,
          isPending: false,
        })
      },
      onError: (error) => {
        updateMessage(conversation.id, assistantId, {
          content: apiErrorMessage(error),
          isPending: false,
          isError: true,
        })
      },
    })
  }

  return (
    <div className="flex h-full flex-col">
      <ChatToolbar
        conversation={conversation}
        collections={collectionsData?.collections ?? []}
        providers={providersData}
        ephemeralFiles={ephemeralFiles}
      />
      <MessageList messages={conversation.messages} />
      <MessageInput onSend={handleSend} disabled={sendMutation.isPending} />
    </div>
  )
}
