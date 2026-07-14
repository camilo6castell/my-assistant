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

type Enhancement = {
  key: "web" | "think" | "agent";
  label: string;
  icon: typeof Globe;
  active: boolean;
  disabled: boolean;
  title: string;
  onToggle: () => void;
  /** El botón "Agente" tiene su propio gradiente -- ver agent-gradient-active en index.css. */
  variant?: "gradient";
};

/**
 * Modo de respuesta de la conversación: SOFT/HARD + las tres "Mejoras"
 * (Web, Pensar, Agente). Vive en el sidebar izquierdo, donde antes
 * estaba el picker de colecciones (que se movió al sidebar derecho,
 * ver RightSidebar.tsx -- ahora agrupa todo lo relacionado con
 * conocimiento: adjuntos, colecciones efímeras, colecciones de
 * sistema). Los ajustes de retrieval (Turnos de historial, Top K) se
 * movieron con las colecciones -- ver AdvancedRetrievalSection.tsx.
 */
export function ResponseModeSection({
  conversation,
  providers,
}: {
  conversation: Conversation;
  providers: ProvidersResponse | undefined;
}) {
  const setMode = useConversationsStore((s) => s.setMode);
  const setGeneration = useConversationsStore((s) => s.setGeneration);
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
  // default_think en ProviderInfo) -- no un "apagado" fijo.
  const effectiveThink =
    conversation.generation.thinkMode ?? activeProvider?.default_think ?? false;

  const enhancements: Enhancement[] = [
    {
      key: "web",
      label: "Web",
      icon: Globe,
      active: conversation.useWebSearch,
      disabled: webSearchQuotaExceeded,
      title: webSearchQuotaExceeded
        ? WEB_SEARCH_QUOTA_EXCEEDED_EXPLANATION
        : WEB_SEARCH_EXPLANATION,
      onToggle: () => setUseWebSearch(conversation.id, !conversation.useWebSearch),
    },
    {
      key: "think",
      label: "Pensar",
      icon: Brain,
      active: effectiveThink,
      disabled: !supportsThinkMode,
      title: supportsThinkMode
        ? THINK_EXPLANATION
        : `El modelo activo (${activeProvider?.model ?? "sin provider"}) no tiene modo de razonamiento configurado.`,
      onToggle: () => setGeneration(conversation.id, { thinkMode: !effectiveThink }),
    },
    {
      key: "agent",
      label: "Agente",
      icon: Bot,
      active: conversation.useAgent,
      disabled: false,
      title: AGENT_EXPLANATION,
      onToggle: () => setUseAgent(conversation.id, !conversation.useAgent),
      variant: "gradient",
    },
  ];

  return (
    <div className="space-y-4 px-3">
      {/* Modo de búsqueda -- control segmentado, mismo lenguaje visual que
          un toggle de tema claro/oscuro: dos estados mutuamente excluyentes. */}
      <div className="space-y-1.5">
        <span className="block text-xs font-medium text-muted-foreground">
          Modo de búsqueda
        </span>
        <div className="grid grid-cols-2 gap-1 rounded-lg border border-white/10 bg-white/[0.03] p-1">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              aria-pressed={conversation.mode === m}
              className={cn(
                "rounded-md px-2.5 py-1.5 text-xs font-medium transition-all",
                conversation.mode === m
                  ? "bg-white/10 text-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-white/5 hover:text-foreground",
              )}
            >
              {m === "SOFT" ? "Suave" : "Estricto"}
            </button>
          ))}
        </div>
      </div>

      {/* Mejoras -- tarjetas compactas en vez de pills en una fila: cada
          una comunica su estado con icono + label + check, en vez de
          depender solo del color de fondo (más legible, más fácil de
          escanear con varias opciones a la vez -- mismo lenguaje que
          Claude.ai/ChatGPT para "Extended thinking"/"Search"). */}
      <div className="space-y-1.5">
        <span className="block text-xs font-medium text-muted-foreground">
          Mejoras
        </span>
        <div className="grid grid-cols-3 gap-1.5">
          {enhancements.map(({ key, label, icon: Icon, active, disabled, title, onToggle, variant }) => (
            <button
              key={key}
              type="button"
              onClick={onToggle}
              disabled={disabled}
              aria-pressed={active}
              title={title}
              className={cn(
                "flex flex-col items-center gap-1 rounded-lg border px-2 py-2 text-[11px] font-medium transition-all",
                disabled
                  ? "cursor-not-allowed border-white/5 text-muted-foreground/30"
                  : active
                    ? variant === "gradient"
                      ? "agent-gradient-active border-transparent text-white shadow-[0_0_12px_rgba(168,85,247,0.45)]"
                      : "border-primary/40 bg-primary/15 text-primary"
                    : "border-white/10 text-muted-foreground hover:bg-white/5 hover:text-foreground",
              )}
            >
              <Icon className="size-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {webSearchQuotaExceeded && (
        <p className="flex items-center justify-between gap-2 text-[10.5px] text-amber-400/80">
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

      {!activeProvider && (
        <p className="text-xs text-muted-foreground">
          No se pudo determinar el provider activo.
        </p>
      )}
    </div>
  );
}
