import { Bot } from "lucide-react"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { Conversation } from "@/types/chat"

export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  const setMode = useConversationsStore((s) => s.setMode)
  const setUseAgent = useConversationsStore((s) => s.setUseAgent)

  return (
    <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5 backdrop-blur-xl">
      <div className="flex flex-wrap items-center gap-1.5">
        {conversation.activeCollections.length === 0 ? (
          <span className="text-xs text-muted-foreground">
            Sin colecciones activas -- elegí en el panel de la izquierda
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            {conversation.activeCollections.length} colección(es) activa(s)
          </span>
        )}
      </div>

      <div className="flex items-center gap-1">
        <div className="mr-1 flex overflow-hidden rounded-lg border border-white/10 text-xs">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              className={cn(
                "px-2.5 py-1.5 transition-colors",
                conversation.mode === m
                  ? "bg-white/10 text-foreground"
                  : "text-muted-foreground hover:bg-white/5"
              )}
            >
              {m}
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={() => setUseAgent(conversation.id, !conversation.useAgent)}
          title="Usar pipeline agente (LangGraph, con revisión)"
          className={cn(
            "flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs transition-colors",
            conversation.useAgent
              ? "border-primary/40 bg-primary/15 text-foreground"
              : "border-white/10 text-muted-foreground hover:bg-white/5"
          )}
        >
          <Bot className="size-3.5" />
          Agente
        </button>
      </div>
    </div>
  )
}
