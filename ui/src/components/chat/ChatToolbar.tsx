import { Wrench } from "lucide-react"
import type { Conversation } from "@/types/chat"

export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  return (
    <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5 backdrop-blur-xl">
      <div className="flex flex-wrap items-center gap-1.5">
        {conversation.taskModeActive ? (
          <span className="flex items-center gap-1.5 text-xs text-primary">
            <Wrench className="size-3.5" />
            Modo Task -- sin colecciones, contexto solo por archivos adjuntos
          </span>
        ) : conversation.activeCollections.length === 0 ? (
          <span className="text-xs text-muted-foreground">
            Sin colecciones activas -- elegí en el panel de la izquierda
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            {conversation.activeCollections.length} colección(es) activa(s)
          </span>
        )}
      </div>
    </div>
  )
}
