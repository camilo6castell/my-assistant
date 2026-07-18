import { Sparkles } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { DEMO_MODE } from "@/lib/demo"
import { useConversationsStore } from "@/stores/conversationsStore"

export function EmptyState() {
  const navigate = useNavigate()
  const createConversation = useConversationsStore((s) => s.createConversation)

  function handleStart() {
    const id = createConversation()
    navigate(`/c/${id}`)
  }

  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl border border-border bg-overlay-strong backdrop-blur-xl">
        <Sparkles className="size-6 text-primary" />
      </div>
      <div className="space-y-1">
        <h1 className="text-lg font-medium text-foreground">Your RAG assistant</h1>
        <p className="max-w-sm text-sm text-muted-foreground">
          {DEMO_MODE
            ? "This is a static portfolio demo -- chat directly with Gemini, no backend or indexed collections behind it."
            : "Pick an existing conversation or start a new one to query your collections."}
        </p>
      </div>
      <Button onClick={handleStart} className="mt-2">
        New conversation
      </Button>
    </div>
  )
}
