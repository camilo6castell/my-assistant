import { Brain, FlaskConical, Globe, Cpu } from "lucide-react"
import { DEMO_MODE, DEMO_MODE_EXPLANATION } from "@/lib/demo"
import { detectSendMode } from "@/lib/sendMessage"
import { cn } from "@/lib/utils"
import { useConversationsStore } from "@/stores/conversationsStore"
import type { ProvidersResponse } from "@/types/api"
import type { Conversation } from "@/types/chat"

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
  key: "web" | "think" | "demo"
  label: string
  icon: typeof Globe
  active: boolean
  disabled: boolean
  title: string
  onToggle: () => void
  variant?: "gradient"
}

export function ResponseModeSection({
  conversation,
  providers,
}: {
  conversation: Conversation
  providers: ProvidersResponse | undefined
}) {
  const currentMode = detectSendMode()
  const setMode = useConversationsStore((s) => s.setMode)
  const setGeneration = useConversationsStore((s) => s.setGeneration)
  const setUseWebSearch = useConversationsStore((s) => s.setUseWebSearch)
  const webSearchQuotaExceeded = useConversationsStore((s) => s.webSearchQuotaExceeded)
  const setWebSearchQuotaExceeded = useConversationsStore((s) => s.setWebSearchQuotaExceeded)

  const activeProvider = providers
    ? providers.providers[providers.active_generation_provider]
    : undefined
  const supportsThinkMode = activeProvider?.supports?.includes("think_mode") ?? false
  const effectiveThink = conversation.generation.thinkMode ?? activeProvider?.default_think ?? false

  const enhancements: Enhancement[] = [
    {
      key: "web",
      label: "Web",
      icon: Globe,
      active: !DEMO_MODE && currentMode !== "client_agent" && conversation.useWebSearch,
      disabled: DEMO_MODE || currentMode === "client_agent" || webSearchQuotaExceeded,
      title: DEMO_MODE
        ? DEMO_MODE_EXPLANATION
        : currentMode === "client_agent"
          ? "Web search is not available when the in-browser agent is active."
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
      {/* Search mode -- segmented control */}
      <div className="space-y-2">
        <span className="block text-[11px] font-medium uppercase tracking-wider text-muted-foreground/60">
          Search mode
        </span>
        <div className="grid grid-cols-2 gap-1 rounded-xl border border-border/60 bg-overlay/50 p-1">
          {(["SOFT", "HARD"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(conversation.id, m)}
              aria-pressed={conversation.mode === m}
              className={cn(
                "rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all duration-150",
                conversation.mode === m
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:bg-overlay-hover hover:text-foreground"
              )}
            >
              {m === "SOFT" ? "Soft" : "Strict"}
            </button>
          ))}
        </div>
      </div>

      {/* Pipeline indicator */}
      {!DEMO_MODE && (
        <div className="flex items-center gap-1.5 rounded-lg border border-border/40 bg-overlay/30 px-2.5 py-1.5">
          <Cpu className="size-3 text-muted-foreground/60" />
          <span className="text-[11px] text-muted-foreground/70">
            {currentMode === "client_agent"
              ? "Full agent pipeline (in-browser)"
              : "Simple pipeline (backend streaming)"}
          </span>
        </div>
      )}

      {/* Enhancements -- compact cards */}
      <div className="space-y-2">
        <span className="block text-[11px] font-medium uppercase tracking-wider text-muted-foreground/60">
          Enhancements
        </span>
        <div className={cn("grid gap-1.5", DEMO_MODE ? "grid-cols-3" : "grid-cols-2")}>
          {enhancements.map(({ key, label, icon: Icon, active, disabled, title, onToggle, variant }) => (
            <button
              key={key}
              type="button"
              onClick={onToggle}
              disabled={disabled}
              aria-pressed={active}
              title={title}
              className={cn(
                "flex flex-col items-center gap-1 rounded-xl border px-2 py-2.5 text-[11px] font-medium transition-all duration-150",
                active
                  ? cn(
                      variant === "gradient"
                        ? "agent-gradient-active border-transparent text-white shadow-[0_0_12px_rgba(168,85,247,0.45)]"
                        : "border-primary/30 bg-primary/8 text-primary",
                      disabled && "cursor-default"
                    )
                  : disabled
                    ? "cursor-not-allowed border-border/30 text-muted-foreground/25"
                    : "border-border/50 text-muted-foreground hover:bg-overlay-hover hover:text-foreground hover:border-border"
              )}
            >
              <Icon className="size-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      {!DEMO_MODE && webSearchQuotaExceeded && (
        <p className="flex items-center justify-between gap-2 text-[10.5px] text-amber-500/80">
          <span>Tavily's quota ran out.</span>
          <button
            type="button"
            onClick={() => setWebSearchQuotaExceeded(false)}
            className="shrink-0 underline decoration-dotted underline-offset-2 hover:text-amber-400 transition-colors"
          >
            Renewed already? Retry
          </button>
        </p>
      )}

      {!DEMO_MODE && !activeProvider && (
        <p className="text-xs text-muted-foreground/60">Couldn't determine the active provider.</p>
      )}
    </div>
  )
}
