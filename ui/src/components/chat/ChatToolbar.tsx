import { Bot } from "lucide-react"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"
import type { useEphemeralFiles } from "@/hooks/useEphemeralFiles"
import { FilesPanel } from "./FilesPanel"
import { GenerationSettingsPopover } from "./GenerationSettingsPopover"

export function ChatToolbar({
  conversation,
  collections,
  providers,
  ephemeralFiles,
}: {
  conversation: Conversation
  collections: string[]
  providers: ProvidersResponse | undefined
  ephemeralFiles: ReturnType<typeof useEphemeralFiles>
}) {
  const setActiveCollections = useConversationsStore((s) => s.setActiveCollections)
  const setMode = useConversationsStore((s) => s.setMode)
  const setUseAgent = useConversationsStore((s) => s.setUseAgent)

  function toggleCollection(name: string) {
    const isActive = conversation.activeCollections.includes(name)
    setActiveCollections(
      conversation.id,
      isActive
        ? conversation.activeCollections.filter((c) => c !== name)
        : [...conversation.activeCollections, name]
    )
  }

  return (
    <div className="relative z-10 flex flex-wrap items-center gap-2 border-b border-white/10 bg-white/[0.02] px-4 py-2.5 backdrop-blur-xl">
      <div className="flex flex-1 flex-wrap items-center gap-1.5">
        {collections.length === 0 && (
          <span className="text-xs text-muted-foreground">Sin colecciones disponibles</span>
        )}
        {collections.map((name) => {
          const isActive = conversation.activeCollections.includes(name)
          return (
            <button
              key={name}
              type="button"
              onClick={() => toggleCollection(name)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs transition-colors",
                isActive
                  ? "border-primary/40 bg-primary/15 text-foreground"
                  : "border-white/10 bg-transparent text-muted-foreground hover:bg-white/5"
              )}
            >
              {name}
            </button>
          )
        })}
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

        <FilesPanel files={ephemeralFiles} />
        <GenerationSettingsPopover conversation={conversation} providers={providers} />
      </div>
    </div>
  )
}
