import { Popover } from "@base-ui/react/popover"
import { Settings2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"
import { useConversationsStore } from "@/stores/conversationsStore"

export function GenerationSettingsPopover({
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
    <Popover.Root>
      <Popover.Trigger
        render={
          <Button variant="ghost" size="icon" aria-label="Opciones de generación">
            <Settings2 className="size-4" />
          </Button>
        }
      />
      <Popover.Portal>
        <Popover.Positioner sideOffset={8} align="end">
          <Popover.Popup className="w-72 rounded-xl border border-white/10 bg-popover/95 p-4 text-sm text-popover-foreground shadow-2xl backdrop-blur-2xl">
            <p className="mb-3 text-xs font-medium text-muted-foreground">
              Opciones de generación
              {activeProvider && (
                <span className="ml-1 text-muted-foreground/70">
                  ({activeProvider.name} · {activeProvider.model})
                </span>
              )}
            </p>

            <div className="space-y-4">
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
                  <span className="text-xs text-muted-foreground">
                    Modo razonamiento (think)
                  </span>
                  <Switch
                    checked={conversation.generation.thinkMode ?? false}
                    onCheckedChange={(checked) =>
                      setGeneration(conversation.id, { thinkMode: checked })
                    }
                  />
                </label>
              )}
            </div>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  )
}
