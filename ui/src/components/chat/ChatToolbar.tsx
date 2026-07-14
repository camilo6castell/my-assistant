import type { Conversation } from "@/types/chat"

export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  const hasContextSource =
    conversation.activeCollections.length > 0 || conversation.useWebSearch

  return (
    <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5 backdrop-blur-xl">
      <div className="flex flex-wrap items-center gap-1.5">
        {hasContextSource ? (
          <span className="text-xs text-muted-foreground">
            {conversation.activeCollections.length > 0 &&
              `${conversation.activeCollections.length} colección(es) activa(s)`}
            {conversation.activeCollections.length > 0 && conversation.useWebSearch && " · "}
            {conversation.useWebSearch && "búsqueda web activa"}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            Sin colecciones activas -- el modelo responde directo, sin RAG
            (podés adjuntar un archivo puntual en el panel de la derecha)
          </span>
        )}
      </div>
    </div>
  )
}
