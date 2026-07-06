import { Switch } from "@/components/ui/switch"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"

export function GenerationSection({
  conversation,
  providers,
}: {
  conversation: Conversation
  providers: ProvidersResponse | undefined
}) {
  const setGeneration = useConversationsStore((s) => s.setGeneration)

  const activeProvider = providers
    ? providers.providers[providers.active_generation_provider]
    : undefined
  const supportsThinkMode = activeProvider?.supports.includes("think_mode") ?? false

  return (
    <div className="space-y-4 px-3">
      {activeProvider && (
        <p className="text-[11px] text-muted-foreground/70">
          {activeProvider.name} · {activeProvider.model}
        </p>
      )}

      <label className="block space-y-1.5">
        <span className="flex items-center justify-between text-xs text-muted-foreground">
          Temperatura
          <span>{conversation.generation.temperature ?? "default"}</span>
        </span>
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          value={conversation.generation.temperature ?? 0.7}
          onChange={(e) =>
            setGeneration(conversation.id, { temperature: Number(e.target.value) })
          }
          className="w-full accent-primary"
        />
      </label>

      {supportsThinkMode && (
        <label className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Modo razonamiento (think)</span>
          <Switch
            checked={conversation.generation.thinkMode ?? false}
            onCheckedChange={(checked) => setGeneration(conversation.id, { thinkMode: checked })}
          />
        </label>
      )}

      {!activeProvider && (
        <p className="text-xs text-muted-foreground">
          No se pudo determinar el provider activo.
        </p>
      )}
    </div>
  )
}
