import { Bot, Brain, FlaskConical, Globe } from "lucide-react"
import { DEMO_MODE, DEMO_MODE_EXPLANATION } from "@/lib/demo"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"

const AGENT_EXPLANATION =
  "Agent mode: evaluates the answer with a second LLM (reviewer) before " +
  "delivering it; if it detects hallucinations or a lack of grounding in " +
  "the context, it asks for a regeneration with feedback. Slower, but " +
  "more reliable."

const WEB_SEARCH_EXPLANATION =
  "Complements the answer with a web search (Tavily). With no active " +
  "collections, the web becomes the only source of context. With active " +
  "collections, it answers with local RAG first and only adds an " +
  "'according to the web...' paragraph if it genuinely adds something new."

const THINK_EXPLANATION =
  "Reasoning (think) mode: the model thinks step by step before " +
  "answering. Slower, but can improve complex answers. Only available " +
  "if the active model supports it."

const WEB_SEARCH_QUOTA_EXCEEDED_EXPLANATION =
  "The Tavily account's quota ran out (free tier or another plan). Check " +
  "your plan at https://app.tavily.com, or wait for the next billing cycle."

type Enhancement = {
  key: "web" | "think" | "agent" | "demo"
  label: string
  icon: typeof Globe
  active: boolean
  disabled: boolean
  title: string
  onToggle: () => void
  /** El botón "Agente" tiene su propio gradiente -- ver agent-gradient-active en index.css. */
  variant?: "gradient"
}

/**
 * Modo de respuesta de la conversación: SOFT/HARD + las "Mejoras" (Web,
 * Pensar, Agente -- + Demo cuando VITE_DEMO_MODE). Vive en el sidebar
 * izquierdo, donde antes estaba el picker de colecciones (que se movió
 * al sidebar derecho, ver RightSidebar.tsx -- ahora agrupa todo lo
 * relacionado con conocimiento: adjuntos, colecciones efímeras,
 * colecciones de sistema). Los ajustes de retrieval (Turnos de
 * historial, Top K) ya no son overrides por-conversación en absoluto --
 * pasaron a ser exclusivamente configuración de servidor (.env), sin
 * ninguna UI (ver GenerationOptions en src/api/schemas/chat.py).
 *
 * En demo mode (ver lib/demo.ts) no hay backend: Web/Pensar/Agente se
 * fuerzan deshabilitados (ninguno funciona sin el backend real) y se
 * agrega un cuarto botón "Demo", siempre activo y sin forma de
 * apagarlo -- es la única "mejora" que tiene sentido ofrecer acá.
 */
export function ResponseModeSection({
  conversation,
  providers,
}: {
  conversation: Conversation
  providers: ProvidersResponse | undefined
}) {
  const setMode = useConversationsStore((s) => s.setMode)
  const setGeneration = useConversationsStore((s) => s.setGeneration)
  const setUseAgent = useConversationsStore((s) => s.setUseAgent)
  const setUseWebSearch = useConversationsStore((s) => s.setUseWebSearch)
  const webSearchQuotaExceeded = useConversationsStore((s) => s.webSearchQuotaExceeded)
  const setWebSearchQuotaExceeded = useConversationsStore((s) => s.setWebSearchQuotaExceeded)

  const activeProvider = providers
    ? providers.providers[providers.active_generation_provider]
    : undefined
  const supportsThinkMode = activeProvider?.supports?.includes("think_mode") ?? false
  // Sin override en esta conversación (null), el efecto real es el
  // default YA escrito en _MODELS[model] para el modelo activo (ver
  // default_think en ProviderInfo) -- no un "apagado" fijo.
  const effectiveThink = conversation.generation.thinkMode ?? activeProvider?.default_think ?? false

  const enhancements: Enhancement[] = [
    {
      key: "web",
      label: "Web",
      icon: Globe,
      active: !DEMO_MODE && conversation.useWebSearch,
      disabled: DEMO_MODE || webSearchQuotaExceeded,
      title: DEMO_MODE
        ? DEMO_MODE_EXPLANATION
        : webSearchQuotaExceeded
          ? WEB_SEARCH_QUOTA_EXCEEDED_EXPLANATION
          : WEB_SEARCH_EXPLANATION,
      onToggle: () => setUseWebSearch(conversation.id, !conversation.useWebSearch),
    },
    {
      key: "think",
      label: "Think",
      icon: Brain,
      active: !DEMO_MODE && effectiveThink,
      disabled: DEMO_MODE || !supportsThinkMode,
      title: DEMO_MODE
        ? DEMO_MODE_EXPLANATION
        : supportsThinkMode
          ? THINK_EXPLANATION
          : `The active model (${activeProvider?.model ?? "no provider"}) doesn't have reasoning mode configured.`,
      onToggle: () => setGeneration(conversation.id, { thinkMode: !effectiveThink }),
    },
    {
      key: "agent",
      label: "Agent",
      icon: Bot,
      active: !DEMO_MODE && conversation.useAgent,
      disabled: DEMO_MODE,
      title: DEMO_MODE ? DEMO_MODE_EXPLANATION : AGENT_EXPLANATION,
      onToggle: () => setUseAgent(conversation.id, !conversation.useAgent),
      // variant: "gradient",
    },
    ...(DEMO_MODE
      ? [
          {
            key: "demo" as const,
            label: "Demo",
            icon: FlaskConical,
            active: true,
            disabled: true,
            title: DEMO_MODE_EXPLANATION,
            onToggle: () => {},
          },
        ]
      : []),
  ]

  return (
    <div className="space-y-4 px-3">
      {/* Modo de búsqueda -- control segmentado, mismo lenguaje visual que
          un toggle de tema claro/oscuro: dos estados mutuamente excluyentes. */}
      <div className="space-y-1.5">
        <span className="block text-xs font-medium text-muted-foreground">Search mode</span>
        <div className="grid grid-cols-2 gap-1 rounded-lg border border-border bg-overlay p-1">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              aria-pressed={conversation.mode === m}
              className={cn(
                "rounded-md px-2.5 py-1.5 text-xs font-medium transition-all",
                conversation.mode === m
                  ? "bg-overlay-strong text-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-overlay-hover hover:text-foreground"
              )}
            >
              {m === "SOFT" ? "Soft" : "Strict"}
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
        <span className="block text-xs font-medium text-muted-foreground">Enhancements</span>
        <div className={cn("grid gap-1.5", DEMO_MODE ? "grid-cols-4" : "grid-cols-3")}>
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
                active
                  ? cn(
                      variant === "gradient"
                        ? "agent-gradient-active border-transparent text-white shadow-[0_0_12px_rgba(168,85,247,0.45)]"
                        : "border-primary/40 bg-primary/15 text-primary",
                      disabled && "cursor-default"
                    )
                  : disabled
                    ? "cursor-not-allowed border-border/50 text-muted-foreground/30"
                    : "border-border text-muted-foreground hover:bg-overlay-hover hover:text-foreground"
              )}
            >
              <Icon className="size-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {!DEMO_MODE && webSearchQuotaExceeded && (
        <p className="flex items-center justify-between gap-2 text-[10.5px] text-amber-400/80">
          <span>Tavily's quota ran out.</span>
          <button
            type="button"
            onClick={() => setWebSearchQuotaExceeded(false)}
            className="shrink-0 underline decoration-dotted underline-offset-2 hover:text-amber-300"
          >
            Renewed already? Retry
          </button>
        </p>
      )}

      {!DEMO_MODE && !activeProvider && (
        <p className="text-xs text-muted-foreground">Couldn't determine the active provider.</p>
      )}
    </div>
  )
}
