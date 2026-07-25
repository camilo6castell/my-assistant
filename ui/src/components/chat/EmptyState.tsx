import { FileText, Sparkles, Search, Globe } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { DEMO_MODE } from "@/lib/demo"
import { useConversationsStore } from "@/stores/conversationsStore"
import { cn } from "@/lib/utils"

const SUGGESTIONS = [
  {
    icon: FileText,
    label: "Ask about my files",
    description: "Query your indexed collections",
  },
  {
    icon: Globe,
    label: "Search the web",
    description: "Get answers with live web context",
  },
  {
    icon: Search,
    label: "Deep research",
    description: "Agent mode with verification",
  },
]

export function EmptyState() {
  const navigate = useNavigate()
  const createConversation = useConversationsStore((s) => s.createConversation)

  function handleStart() {
    const id = createConversation()
    navigate(`/c/${id}`)
  }

  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center px-6 fade-in">
      <div className="flex flex-col items-center gap-6 max-w-md text-center">
        {/* Logo mark */}
        <div className="relative">
          <div className="flex size-16 items-center justify-center rounded-2xl bg-primary/8 border border-primary/10">
            <Sparkles className="size-7 text-primary/70" />
          </div>
        </div>

        {/* Heading */}
        <div className="space-y-2">
          <h1 className="text-xl font-semibold tracking-tight text-foreground">
            Your RAG assistant
          </h1>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {DEMO_MODE
              ? "This is a static portfolio demo. Chat directly with Gemini, no backend or indexed collections behind it."
              : "Start a conversation to query your collections, attach files, or search the web."}
          </p>
        </div>

        {/* CTA */}
        <Button onClick={handleStart} size="lg" className="rounded-xl px-6">
          New conversation
        </Button>

        {/* Quick suggestions */}
        {!DEMO_MODE && (
          <div className="grid grid-cols-3 gap-3 w-full mt-2">
            {SUGGESTIONS.map(({ icon: Icon, label, description }) => (
              <button
                key={label}
                type="button"
                onClick={handleStart}
                className={cn(
                  "flex flex-col items-center gap-2 rounded-xl border border-border p-4 text-center",
                  "transition-all duration-150",
                  "hover:border-primary/20 hover:bg-primary/5 hover:shadow-sm",
                  "cursor-pointer",
                )}
              >
                <Icon className="size-4 text-muted-foreground" />
                <span className="text-xs font-medium text-foreground leading-tight">
                  {label}
                </span>
                <span className="text-[10px] text-muted-foreground/60 leading-tight">
                  {description}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
