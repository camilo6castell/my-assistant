import { useMutation, useQueryClient } from "@tanstack/react-query"
import { nanoid } from "nanoid"
import { Navigate, useParams } from "react-router-dom"
import { apiErrorMessage, isWebSearchQuotaExceededError, postQuery } from "@/lib/api/client"
import { askGeminiDemo, buildAttachmentsContext, DEMO_MODE, type DemoAnswer } from "@/lib/demo"
import { useConversationsStore } from "@/stores/conversationsStore"
import { useDemoAttachmentsStore, selectDemoFiles } from "@/stores/demoAttachmentsStore"
import type { ChatMessage } from "@/types/chat"
import type { QueryResponse } from "@/types/api"
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
  const deleteMessage = useConversationsStore((s) => s.deleteMessage)
  const setWebSearchQuotaExceeded = useConversationsStore((s) => s.setWebSearchQuotaExceeded)
  const queryClient = useQueryClient()
  const demoAttachments = useDemoAttachmentsStore(selectDemoFiles(conversation?.id ?? null))
  const clearDemoAttachments = useDemoAttachmentsStore((s) => s.clearFiles)

  // Un solo camino de envío para toda conversación -- /query decide
  // internamente entre raw/web-only/RAG según haya o no colecciones/
  // archivos efímeros/web_search (ver _answer_raw en
  // src/api/routers/chat.py). Sin colecciones activas ni web_search, el
  // usuario puede seguir preguntando: la pregunta (más los archivos
  // adjuntos, si los hay) va directo al LLM sin ningún system prompt.
  //
  // En VITE_DEMO_MODE no hay backend: se llama directo a Gemini desde el
  // navegador (ver lib/demo.ts) con los adjuntos demo (si hay) inyectados
  // como contexto -- misma semántica que los adjuntos reales, solo que el
  // "backend" que los inyecta es esta función.
  const sendMutation = useMutation<QueryResponse | DemoAnswer, unknown, string>({
    mutationFn: (text: string) => {
      if (!conversation) throw new Error("Conversation not found")

      if (DEMO_MODE) {
        return askGeminiDemo({
          question: text,
          history: toHistory(conversation.messages),
          attachmentsContext: buildAttachmentsContext(demoAttachments),
        })
      }

      return postQuery(
        {
          question: text,
          collections: conversation.activeCollections,
          mode: conversation.mode,
          chat_history: toHistory(conversation.messages),
          conversation_id: conversation.id,
          generation: {
            max_tokens: conversation.generation.maxTokens,
            think_mode: conversation.generation.thinkMode,
          },
          web_search: conversation.useWebSearch,
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
      pendingLabel: !DEMO_MODE && conversation.useWebSearch ? "Checking the web..." : undefined,
    }

    addMessage(conversation.id, userMsg)
    addMessage(conversation.id, pendingMsg)

    sendMutation.mutate(text, {
      onSuccess: (data) => {
        if ("isDemo" in data) {
          updateMessage(conversation.id, assistantId, {
            content: data.answer,
            isPending: false,
          })
          // Espejo de _consume_attachments en el backend real: se
          // consumen (borran) al enviarlos exitosamente.
          clearDemoAttachments(conversation.id)
          return
        }

        updateMessage(conversation.id, assistantId, {
          content: data.answer,
          confidence: data.confidence,
          collectionsUsed: data.collections_used,
          reformulated: data.reformulated,
          usedWebSearch: data.used_web_search,
          webSources: data.web_sources ?? undefined,
          isPending: false,
        })
        // Caso B silencioso (ver _supplement_with_web en el backend): la
        // respuesta principal se generó igual, pero el complemento web
        // se omitió por cuota agotada -- el mensaje no lo refleja, así
        // que esta es la única señal.
        if (data.web_search_quota_exceeded) {
          setWebSearchQuotaExceeded(true)
        }
        // Los archivos adjuntos, si había, se consumieron en el backend
        // al procesar esta query (ver _consume_attachments en
        // src/api/routers/chat.py) -- se refresca la lista para que el
        // sidebar derecho refleje que ya no están adjuntos.
        queryClient.invalidateQueries({ queryKey: ["attachments", conversation.id] })
      },
      onError: (error) => {
        updateMessage(conversation.id, assistantId, {
          content: apiErrorMessage(error),
          isPending: false,
          isError: true,
        })
        if (DEMO_MODE) return
        // Caso A (ver _answer_web_only en el backend): sin colecciones,
        // la búsqueda web falló por cuota agotada y el 422 lo comunica
        // en el detail estructurado.
        if (isWebSearchQuotaExceededError(error)) {
          setWebSearchQuotaExceeded(true)
        }
        // 413 del guard de contexto (src/llm/context_guard.py) llega acá
        // también: apiErrorMessage() ya arma el mensaje legible (tokens
        // estimados/límite) a partir del detail estructurado.
      },
    })
  }

  function handleDeleteMessage(messageId: string) {
    if (!conversation) return
    deleteMessage(conversation.id, messageId)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <ChatToolbar conversation={conversation} />
      <MessageList messages={conversation.messages} onDeleteMessage={handleDeleteMessage} />
      <MessageInput onSend={handleSend} disabled={sendMutation.isPending} />
    </div>
  )
}
