import { Bot, Brain, Globe } from "lucide-react";
import { cn } from "@/lib/utils";
import { useConversationsStore } from "@/stores/conversationsStore";
import type { ProvidersResponse } from "@/types/api";
import type { Conversation } from "@/types/chat";

const AGENT_EXPLANATION =
  "Modo agente: evalúa la respuesta con un segundo LLM (reviewer) antes de " +
  "entregarla; si detecta alucinaciones o falta de anclaje al contexto, pide " +
  "regenerarla con feedback. Más lento, pero más confiable.";

const WEB_SEARCH_EXPLANATION =
  "Complementa la respuesta con una búsqueda web (Tavily). Sin colecciones " +
  "activas, la web pasa a ser la única fuente de contexto. Con colecciones " +
  "activas, primero responde con el RAG local y solo agrega un párrafo " +
  "'según la web...' si de verdad aporta algo nuevo.";

const THINK_EXPLANATION =
  "Modo razonamiento (think): el modelo piensa paso a paso antes de " +
  "responder. Más lento, pero puede mejorar respuestas complejas. Solo " +
  "disponible si el modelo activo lo soporta.";

const WEB_SEARCH_QUOTA_EXCEEDED_EXPLANATION =
  "Se agotó la cuota de la cuenta de Tavily (free tier u otro plan). Revisá " +
  "tu plan en https://app.tavily.com, o esperá al próximo ciclo de facturación.";

/** Parsea un input numérico controlado -- undefined si no es un número > 0 válido. */
function parsePositiveInt(raw: string): number | undefined {
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) return undefined;
  return Math.round(value);
}

export function GenerationSection({
  conversation,
  providers,
}: {
  conversation: Conversation;
  providers: ProvidersResponse | undefined;
}) {
  const setGeneration = useConversationsStore((s) => s.setGeneration);
  const setMode = useConversationsStore((s) => s.setMode);
  const setUseAgent = useConversationsStore((s) => s.setUseAgent);
  const setUseWebSearch = useConversationsStore((s) => s.setUseWebSearch);
  const webSearchQuotaExceeded = useConversationsStore(
    (s) => s.webSearchQuotaExceeded,
  );
  const setWebSearchQuotaExceeded = useConversationsStore(
    (s) => s.setWebSearchQuotaExceeded,
  );

  const activeProvider = providers
    ? providers.providers[providers.active_generation_provider]
    : undefined;
  const supportsThinkMode =
    activeProvider?.supports?.includes("think_mode") ?? false;
  // Sin override en esta conversación (null), el efecto real es el
  // default YA escrito en _MODELS[model] para el modelo activo (ver
  // default_think en ProviderInfo) -- no un "apagado" fijo. El botón
  // tiene que arrancar reflejando ESO, no un estado inventado por la UI
  // que no tiene nada que ver con lo que el backend realmente manda.
  const effectiveThink =
    conversation.generation.thinkMode ?? activeProvider?.default_think ?? false;
  const defaults = providers?.defaults;

  // Defaults de retrieval dependientes del modo SOFT/HARD actual -- mismo
  // criterio que _resolve_top_k en src/api/routers/chat.py.
  const modeTopKDefaults =
    conversation.mode === "SOFT"
      ? {
          initial: defaults?.soft_top_k_initial,
          final: defaults?.soft_top_k_final,
        }
      : {
          initial: defaults?.hard_top_k_initial,
          final: defaults?.hard_top_k_final,
        };

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
    setGeneration(conversation.id, {
      topKFinal: Math.min(value, effectiveTopKInitial),
    });
  }

  function handleMaxTurnsChange(raw: string) {
    const value = parsePositiveInt(raw);
    if (value === undefined) return;
    setGeneration(conversation.id, { maxTurns: value });
  }

  return (
    <div className="flex flex-col justify-between space-y-3 px-3">
      {/* 1. SOFT / HARD */}
      <div className="flex items-center justify-between gap-2">
        <span className="block text-xs text-muted-foreground">
          Modo de búsqueda
        </span>
        <div className="flex overflow-hidden rounded-lg border border-white/10 text-xs">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              className={cn(
                "flex-1 px-2.5 py-1 transition-colors",
                conversation.mode === m
                  ? "bg-white/10 text-foreground"
                  : "text-muted-foreground hover:bg-white/5",
              )}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Modo agente + búsqueda web */}
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
          Mejoras
        </span>
        <div className="flex items-center justify-center gap-1">
          <button
            type="button"
            onClick={() =>
              setUseWebSearch(conversation.id, !conversation.useWebSearch)
            }
            disabled={webSearchQuotaExceeded}
            aria-pressed={conversation.useWebSearch}
            title={
              webSearchQuotaExceeded
                ? WEB_SEARCH_QUOTA_EXCEEDED_EXPLANATION
                : WEB_SEARCH_EXPLANATION
            }
            className={cn(
              "flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
              webSearchQuotaExceeded
                ? "cursor-not-allowed border-white/5 text-muted-foreground/40"
                : conversation.useWebSearch
                  ? "border-primary/40 bg-primary/15 text-primary"
                  : "border-white/10 text-muted-foreground hover:bg-white/5",
            )}
          >
            <Globe className="size-3.5" />
            Web
          </button>
          <button
            type="button"
            onClick={() =>
              setGeneration(conversation.id, { thinkMode: !effectiveThink })
            }
            disabled={!supportsThinkMode}
            aria-pressed={effectiveThink}
            title={
              supportsThinkMode
                ? THINK_EXPLANATION
                : `El modelo activo (${activeProvider?.model ?? "sin provider"}) no tiene modo de razonamiento configurado.`
            }
            className={cn(
              "flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
              !supportsThinkMode
                ? "cursor-not-allowed border-white/5 text-muted-foreground/40"
                : effectiveThink
                  ? "border-primary/40 bg-primary/15 text-primary"
                  : "border-white/10 text-muted-foreground hover:bg-white/5",
            )}
          >
            <Brain className="size-3.5" />
            Pensar
          </button>
          <button
            type="button"
            onClick={() => setUseAgent(conversation.id, !conversation.useAgent)}
            aria-pressed={conversation.useAgent}
            title={AGENT_EXPLANATION}
            className={cn(
              "relative flex items-center gap-1 overflow-hidden rounded-full border px-3 py-1 text-xs font-medium transition-all",
              conversation.useAgent
                ? "agent-gradient-active border-transparent text-white shadow-[0_0_12px_rgba(168,85,247,0.45)]"
                : "border-white/10 text-muted-foreground hover:bg-white/5",
            )}
          >
            <Bot className="size-3.5" />
            Agente
            {/* {conversation.useAgent ? "Agente:  on" : "Agente: off"} */}
          </button>
        </div>
      </div>

      {webSearchQuotaExceeded && (
        <p className="-mt-2 flex items-center justify-between gap-2 text-[10.5px] text-amber-400/80">
          <span>Se agotó la cuota de Tavily.</span>
          <button
            type="button"
            onClick={() => setWebSearchQuotaExceeded(false)}
            className="shrink-0 underline decoration-dotted underline-offset-2 hover:text-amber-300"
          >
            ¿Ya se renovó? Reintentar
          </button>
        </p>
      )}

      {/* 3. Cantidad de turnos */}
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

      {/* 4. Top K inicial / final */}
      <div className="flex items-center justify-between gap-2">
        <span className="fgrow-1 items-center justify-between text-xs text-muted-foreground">
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
      <p className="text-[10.5px] text-muted-foreground/60 leading-[1]">
        *Top K final no puede superar a Top K inicial (se ajusta
        automáticamente).
      </p>

      {!activeProvider && (
        <p className="text-xs text-muted-foreground">
          No se pudo determinar el provider activo.
        </p>
      )}
    </div>
  );
}
