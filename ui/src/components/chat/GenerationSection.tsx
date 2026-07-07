import { Bot } from "lucide-react"
import { InfoTooltip } from "@/components/ui/info-tooltip"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"

const AGENT_EXPLANATION =
  "El modo agente evalúa la respuesta con un segundo LLM (reviewer) antes de " +
  "entregarla: si detecta alucinaciones o falta de anclaje al contexto, pide " +
  "regenerarla con feedback. Más lento, pero más confiable."

/** Parsea un input numérico controlado -- undefined si no es un número > 0 válido. */
function parsePositiveInt(raw: string): number | undefined {
  const value = Number(raw)
  if (!Number.isFinite(value) || value <= 0) return undefined
  return Math.round(value)
}

export function GenerationSection({
  conversation,
  providers,
}: {
  conversation: Conversation
  providers: ProvidersResponse | undefined
}) {
  const setGeneration = useConversationsStore((s) => s.setGeneration)
  const setMode = useConversationsStore((s) => s.setMode)
  const setUseAgent = useConversationsStore((s) => s.setUseAgent)

  const activeProvider = providers
    ? providers.providers[providers.active_generation_provider]
    : undefined
  const supportsThinkMode = activeProvider?.supports.includes("think_mode") ?? false
  const defaults = providers?.defaults

  // Defaults de retrieval dependientes del modo SOFT/HARD actual -- mismo
  // criterio que _resolve_top_k en src/api/routers/chat.py.
  const modeTopKDefaults =
    conversation.mode === "SOFT"
      ? { initial: defaults?.soft_top_k_initial, final: defaults?.soft_top_k_final }
      : { initial: defaults?.hard_top_k_initial, final: defaults?.hard_top_k_final }

  const effectiveTopKInitial =
    conversation.generation.topKInitial ?? modeTopKDefaults.initial ?? 0
  const effectiveTopKFinal = conversation.generation.topKFinal ?? modeTopKDefaults.final ?? 0

  function handleTopKInitialChange(raw: string) {
    const value = parsePositiveInt(raw)
    if (value === undefined) return
    // Si el nuevo initial queda por debajo del final vigente, el final baja
    // con él -- evita quedar en un estado inválido (final > initial) que el
    // backend rechazaría con 400 (ver _resolve_top_k).
    if (effectiveTopKFinal > value) {
      setGeneration(conversation.id, { topKInitial: value, topKFinal: value })
    } else {
      setGeneration(conversation.id, { topKInitial: value })
    }
  }

  function handleTopKFinalChange(raw: string) {
    const value = parsePositiveInt(raw)
    if (value === undefined) return
    setGeneration(conversation.id, { topKFinal: Math.min(value, effectiveTopKInitial) })
  }

  function handleMaxTurnsChange(raw: string) {
    const value = parsePositiveInt(raw)
    if (value === undefined) return
    setGeneration(conversation.id, { maxTurns: value })
  }

  return (
    <div className="space-y-4 px-3">
      {activeProvider && (
        <p className="text-[11px] text-muted-foreground/70">
          {activeProvider.name} · {activeProvider.model}
        </p>
      )}

      {/* 1. SOFT / HARD */}
      <div className="space-y-1.5">
        <span className="block text-xs text-muted-foreground">Modo de búsqueda</span>
        <div className="flex overflow-hidden rounded-lg border border-white/10 text-xs">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              className={cn(
                "flex-1 px-2.5 py-1.5 transition-colors",
                conversation.mode === m
                  ? "bg-white/10 text-foreground"
                  : "text-muted-foreground hover:bg-white/5"
              )}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Modo agente */}
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Bot className="size-3.5" />
          Agente
          <InfoTooltip text={AGENT_EXPLANATION} />
        </span>
        <button
          type="button"
          onClick={() => setUseAgent(conversation.id, !conversation.useAgent)}
          aria-pressed={conversation.useAgent}
          title="Usar pipeline agente (LangGraph, con revisión)"
          className={cn(
            "relative flex items-center gap-1 overflow-hidden rounded-full border px-3 py-1 text-xs font-medium transition-all",
            conversation.useAgent
              ? "agent-gradient-active border-transparent text-white shadow-[0_0_12px_rgba(168,85,247,0.45)]"
              : "border-white/10 text-muted-foreground hover:bg-white/5"
          )}
        >
          {conversation.useAgent ? "Activado" : "Desactivado"}
        </button>
      </div>

      {/* 3. Modo razonamiento */}
      {supportsThinkMode && (
        <label className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Modo razonamiento (think)</span>
          <Switch
            checked={conversation.generation.thinkMode ?? false}
            onCheckedChange={(checked) => setGeneration(conversation.id, { thinkMode: checked })}
          />
        </label>
      )}

      {/* 4. Cantidad de turnos */}
      <label className="block space-y-1.5">
        <span className="flex items-center justify-between text-xs text-muted-foreground">
          Turnos de historial
          <span className="text-muted-foreground/60">
            {defaults ? `.env: ${defaults.max_turns}` : ""}
          </span>
        </span>
        <input
          type="number"
          min={1}
          step={1}
          value={conversation.generation.maxTurns ?? defaults?.max_turns ?? ""}
          onChange={(e) => handleMaxTurnsChange(e.target.value)}
          className="w-full rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
        />
      </label>

      {/* 5. Top K inicial / final */}
      <div className="space-y-1.5">
        <span className="flex items-center justify-between text-xs text-muted-foreground">
          Retrieval ({conversation.mode})
          <span className="text-muted-foreground/60">
            {defaults ? `.env: ${modeTopKDefaults.initial} / ${modeTopKDefaults.final}` : ""}
          </span>
        </span>
        <div className="flex items-center gap-2">
          <label className="flex-1 space-y-1">
            <span className="block text-[11px] text-muted-foreground/70">Top K inicial</span>
            <input
              type="number"
              min={1}
              step={1}
              value={effectiveTopKInitial || ""}
              onChange={(e) => handleTopKInitialChange(e.target.value)}
              className="w-full rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
            />
          </label>
          <label className="flex-1 space-y-1">
            <span className="block text-[11px] text-muted-foreground/70">Top K final</span>
            <input
              type="number"
              min={1}
              max={effectiveTopKInitial || undefined}
              step={1}
              value={effectiveTopKFinal || ""}
              onChange={(e) => handleTopKFinalChange(e.target.value)}
              className="w-full rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
            />
          </label>
        </div>
        <p className="text-[10.5px] text-muted-foreground/60">
          Top K final no puede superar a Top K inicial -- se ajusta automáticamente.
        </p>
      </div>

      {/* 6. Temperatura */}
      <label className="block space-y-1.5">
        <span className="flex items-center justify-between text-xs text-muted-foreground">
          Temperatura
          <span>{conversation.generation.temperature ?? defaults?.temperature ?? "default"}</span>
        </span>
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          value={conversation.generation.temperature ?? defaults?.temperature ?? 0.7}
          onChange={(e) =>
            setGeneration(conversation.id, { temperature: Number(e.target.value) })
          }
          className="w-full accent-primary"
        />
      </label>

      {!activeProvider && (
        <p className="text-xs text-muted-foreground">
          No se pudo determinar el provider activo.
        </p>
      )}
    </div>
  )
}
