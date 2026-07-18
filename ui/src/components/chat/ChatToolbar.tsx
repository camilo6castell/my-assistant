import type { Conversation } from "@/types/chat"
import { DEMO_MODE } from "@/lib/demo"

export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  const hasContextSource =
    conversation.activeCollections.length > 0 || conversation.useWebSearch

  return (
    <div className="relative z-10 flex flex-wrap items-center justify-between gap-2 border-b border-border bg-overlay px-4 py-2.5 backdrop-blur-xl">
      <div className="flex flex-wrap items-center gap-1.5">
        {DEMO_MODE ? (
          <span className="text-xs text-muted-foreground">
            Demo mode -- chatting directly with Gemini, no retrieval behind it
          </span>
        ) : hasContextSource ? (
          <span className="text-xs text-muted-foreground">
            {conversation.activeCollections.length > 0 &&
              `${conversation.activeCollections.length} active collection(s)`}
            {conversation.activeCollections.length > 0 && conversation.useWebSearch && " · "}
            {conversation.useWebSearch && "web search active"}
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">
            No active collections -- the model answers directly, without RAG
            (you can attach a one-off file in the right-hand panel)
          </span>
        )}
      </div>
    </div>
  )
}
