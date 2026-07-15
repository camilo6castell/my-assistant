import { AlertCircle, Globe, Layers, Sparkles, Trash2 } from "lucide-react"
import ReactMarkdown from "react-markdown"
import rehypeHighlight from "rehype-highlight"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import type { ChatMessage } from "@/types/chat"

function ThinkingDots({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      {label && <span className="text-xs text-muted-foreground">{label}</span>}
      <span className="inline-flex items-center gap-1">
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
        <span className="size-1.5 animate-bounce rounded-full bg-muted-foreground" />
      </span>
    </span>
  )
}

export function MessageBubble({
  message,
  onDelete,
}: {
  message: ChatMessage
  /** Ausente = no se puede borrar este mensaje individualmente (no usado hoy, pero deja la puerta abierta). */
  onDelete?: () => void
}) {
  const isUser = message.role === "user"

  return (
    <div className={cn("group flex w-full items-start gap-1.5", isUser ? "justify-end" : "justify-start")}>
      {/* Botón de borrar del lado del avatar/margen -- del lado izquierdo
          para mensajes de usuario (que están alineados a la derecha) y
          del lado derecho para mensajes del assistant, para no quedar
          nunca pegado al texto de la burbuja. Oculto hasta hover de la
          fila (mismo patrón que los botones de renombrar/borrar chat en
          LeftSidebar.tsx) para no ensuciar la lectura normal. */}
      {onDelete && !message.isPending && isUser && (
        <button
          type="button"
          onClick={onDelete}
          aria-label="Delete message"
          title="Delete message"
          className="order-first mt-2.5 shrink-0 self-start rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/10 hover:text-destructive group-hover:opacity-100"
        >
          <Trash2 className="size-3.5" />
        </button>
      )}
      <div
        className={cn(
          "max-w-[75ch] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary/90 text-primary-foreground"
            : "border border-white/10 bg-white/[0.04] text-foreground backdrop-blur-xl",
          message.isError && "border-destructive/30 bg-destructive/10 text-destructive"
        )}
      >
        {message.isPending ? (
          <ThinkingDots label={message.pendingLabel} />
        ) : isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-black/40">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && !message.isPending && !message.isError && (
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-white/10 pt-2 text-[11px] text-muted-foreground">
            {message.confidence !== undefined && (
              <span className="inline-flex items-center gap-1">
                <Sparkles className="size-3" />
                confidence {(message.confidence * 100).toFixed(0)}%
              </span>
            )}
            {!!message.collectionsUsed?.length && (
              <span className="inline-flex items-center gap-1">
                <Layers className="size-3" />
                {message.collectionsUsed.join(", ")}
              </span>
            )}
            {message.usedWebSearch && (
              <span className="inline-flex items-center gap-1">
                <Globe className="size-3" />
                includes web search
              </span>
            )}
            {message.reformulated && <span>· question reformulated</span>}
          </div>
        )}

        {!!message.webSources?.length && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.webSources.map((source) => (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                title={source.url}
                className="inline-flex max-w-[220px] items-center gap-1 truncate rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-white/10 hover:text-foreground"
              >
                <Globe className="size-3 shrink-0" />
                <span className="truncate">{source.title}</span>
              </a>
            ))}
          </div>
        )}

        {message.isError && (
          <div className="mt-2 flex items-center gap-1.5 text-xs">
            <AlertCircle className="size-3.5" />
            Couldn't complete the response
          </div>
        )}
      </div>

      {onDelete && !message.isPending && !isUser && (
        <button
          type="button"
          onClick={onDelete}
          aria-label="Delete message"
          title="Delete message"
          className="mt-2.5 shrink-0 self-start rounded-md p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-white/10 hover:text-destructive group-hover:opacity-100"
        >
          <Trash2 className="size-3.5" />
        </button>
      )}
    </div>
  )
}
