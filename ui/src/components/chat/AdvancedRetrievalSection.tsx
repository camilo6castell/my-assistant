import { cn } from "@/lib/utils";
import { useConversationsStore } from "@/stores/conversationsStore";
import type { ProvidersResponse } from "@/types/api";
import type { Conversation } from "@/types/chat";

/** Parsea un input numérico controlado -- undefined si no es un número > 0 válido. */
function parsePositiveInt(raw: string): number | undefined {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) return undefined;
  return Math.round(value);
}

/**
 * Ajustes finos de retrieval: cuántos turnos de historial se mandan al
 * LLM, y cuántos chunks se recuperan (Top K inicial/final). Vive en el
 * sidebar derecho, debajo de las colecciones -- son parámetros de CÓMO
 * se usa el conocimiento (archivos/colecciones), no del modo de
 * respuesta en sí (eso vive en ResponseModeSection.tsx, sidebar
 * izquierdo). Top K no tiene efecto si no hay colecciones activas (ver
 * _answer_raw en src/api/routers/chat.py), pero se deja siempre
 * disponible -- no vale la pena la complejidad de deshabilitarlo
 * condicionalmente por algo sin costo real si se ignora.
 */
export function AdvancedRetrievalSection({
  conversation,
  providers,
}: {
  conversation: Conversation;
  providers: ProvidersResponse | undefined;
}) {
  const setGeneration = useConversationsStore((s) => s.setGeneration);
  const defaults = providers?.defaults;

  // Defaults de retrieval dependientes del modo SOFT/HARD actual -- mismo
  // criterio que _resolve_top_k en src/api/routers/chat.py.
  const modeTopKDefaults =
    conversation.mode === "SOFT"
      ? { initial: defaults?.soft_top_k_initial, final: defaults?.soft_top_k_final }
      : { initial: defaults?.hard_top_k_initial, final: defaults?.hard_top_k_final };

  const effectiveTopKInitial =
    conversation.generation.topKInitial ?? modeTopKDefaults.initial ?? 0;
  const effectiveTopKFinal =
    conversation.generation.topKFinal ?? modeTopKDefaults.final ?? 0;

  function handleTopKInitialChange(raw: string) {
    const value = parsePositiveInt(raw);
    if (value === undefined) return;
    // Si el nuevo initial queda por debajo del final vigente, el final baja
    // con él -- evita quedar en un estado inválido (final > initial) que el
    // backend rechazaría con 400 (ver _resolve_top_k).
    if (effectiveTopKFinal > value) {
      setGeneration(conversation.id, { topKInitial: value, topKFinal: value });
    } else {
      setGeneration(conversation.id, { topKInitial: value });
    }
  }

  function handleTopKFinalChange(raw: string) {
    const value = parsePositiveInt(raw);
    if (value === undefined) return;
    setGeneration(conversation.id, { topKFinal: Math.min(value, effectiveTopKInitial) });
  }

  function handleMaxTurnsChange(raw: string) {
    const value = parsePositiveInt(raw);
    if (value === undefined) return;
    setGeneration(conversation.id, { maxTurns: value });
  }

  return (
    <div className="space-y-3 px-3">
      {/* Turnos de historial */}
      <label className="flex items-center justify-between gap-2">
        <span className="grow-1 items-end justify-between text-xs text-muted-foreground">
          <span className="block">Turnos de historial</span>
          <span className="text-muted-foreground/60">
            {defaults ? `Default: ${defaults.max_turns}` : ""}
          </span>
        </span>
        <input
          type="number"
          min={1}
          step={1}
          value={conversation.generation.maxTurns ?? defaults?.max_turns ?? ""}
          onChange={(e) => handleMaxTurnsChange(e.target.value)}
          className="w-15 rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
        />
      </label>

      {/* Top K inicial / final */}
      <div className="flex items-center justify-between gap-2">
        <span className={cn("items-center justify-between text-xs text-muted-foreground")}>
          <span className="block">Retrieval ({conversation.mode})</span>
          <span className="text-muted-foreground/60">
            {defaults
              ? `Default: ${modeTopKDefaults.initial} / ${modeTopKDefaults.final}`
              : ""}
          </span>
        </span>
        <div className="flex items-center gap-2">
          <label className="flex-1">
            <span className="block text-[11px] text-muted-foreground/70">
              Top K inicial
            </span>
            <input
              type="number"
              min={1}
              step={1}
              value={effectiveTopKInitial || ""}
              onChange={(e) => handleTopKInitialChange(e.target.value)}
              className="w-15 rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
            />
          </label>
          <label className="flex-1">
            <span className="block text-[11px] text-muted-foreground/70">
              Top K final
            </span>
            <input
              type="number"
              min={1}
              max={effectiveTopKInitial || undefined}
              step={1}
              value={effectiveTopKFinal || ""}
              onChange={(e) => handleTopKFinalChange(e.target.value)}
              className="w-15 rounded-md border border-white/10 bg-white/[0.04] px-2 py-1 text-xs text-foreground outline-none focus:border-primary/50"
            />
          </label>
        </div>
      </div>
      <p className="text-[10.5px] leading-[1] text-muted-foreground/60">
        *Top K final no puede superar a Top K inicial (se ajusta
        automáticamente). Sin colecciones activas, estos valores no
        tienen efecto.
      </p>
    </div>
  );
}
