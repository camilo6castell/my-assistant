import { AlertCircle, Check, Copy, Globe, Layers, Sparkles, Trash2 } from "lucide-react"
import { useState, type ComponentProps } from "react"
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

/**
 * Botón compartido para "copiar" (mensaje completo o bloque de código):
 * icono que rota a un check por 1.5s como confirmación, sin depender de
 * un toast externo -- consistente con el resto de la UI, que ya evita
 * dependencias extra para micro-feedback (ver ThemeToggle, Skeleton).
 */
function CopyButton({
  getText,
  className,
  label = "Copy",
}: {
  getText: () => string
  className?: string
  label?: string
}) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(getText())
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard API puede fallar (permisos, contexto no seguro) --
      // silencioso, no vale la pena un mensaje de error para esto.
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? "Copied" : label}
      title={copied ? "Copied!" : label}
      className={cn(
        "inline-flex items-center justify-center rounded-md p-1 text-muted-foreground transition-colors hover:bg-overlay-hover hover:text-foreground",
        className
      )}
    >
      {copied ? <Check className="size-3.5 text-emerald-500" /> : <Copy className="size-3.5" />}
    </button>
  )
}

/**
 * Reemplaza el <pre> que genera ReactMarkdown/rehype-highlight por una
 * versión con botón de copiar propio -- visible siempre en mobile (no
 * hay hover táctil) y solo al pasar el mouse en desktop.
 */
function CodeBlock({ children, className, ...props }: ComponentProps<"pre">) {
  return (
    <div className="group/code relative">
      <CopyButton
        getText={() => extractText(children)}
        label="Copy code"
        className="absolute right-2 top-2 z-10 bg-overlay-strong opacity-70 backdrop-blur-sm lg:opacity-0 lg:group-hover/code:opacity-100"
      />
      <pre className={className} {...props}>
        {children}
      </pre>
    </div>
  )
}

/** Extrae el texto plano de los children de React (para copiar el código sin markup de highlight.js). */
function extractText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node)
  if (Array.isArray(node)) return node.map(extractText).join("")
  if (node && typeof node === "object" && "props" in node) {
    return extractText((node as { props: { children?: React.ReactNode } }).props.children)
  }
  return ""
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
  const showActions = !message.isPending && !message.isError

  const actions = (
    <span
      className={cn(
        "mt-2.5 flex shrink-0 items-center gap-0.5 self-start opacity-0 transition-opacity group-hover:opacity-100",
        "max-lg:opacity-100" // en mobile no hay hover: siempre visibles
      )}
    >
      {showActions && <CopyButton getText={() => message.content} label="Copy message" />}
      {onDelete && showActions && (
        <button
          type="button"
          onClick={onDelete}
          aria-label="Delete message"
          title="Delete message"
          className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-overlay-hover hover:text-destructive"
        >
          <Trash2 className="size-3.5" />
        </button>
      )}
    </span>
  )

  return (
    <div className={cn("group flex w-full items-start gap-1.5", isUser ? "justify-end" : "justify-start")}>
      {isUser && actions}
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed sm:max-w-[75ch]",
          isUser
            ? "bg-primary/90 text-primary-foreground"
            : "border border-border bg-overlay text-foreground backdrop-blur-xl",
          message.isError && "border-destructive/30 bg-destructive/10 text-destructive"
        )}
      >
        {message.isPending ? (
          <ThinkingDots label={message.pendingLabel} />
        ) : isUser ? (
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        ) : (
          <div className="prose prose-sm dark:prose-invert max-w-none break-words prose-p:leading-relaxed prose-pre:bg-transparent prose-pre:p-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{ pre: CodeBlock }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {!isUser && !message.isPending && !message.isError && (
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border pt-2 text-[11px] text-muted-foreground">
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
                className="inline-flex max-w-[220px] items-center gap-1 truncate rounded-full border border-border bg-overlay px-2 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-overlay-hover hover:text-foreground"
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
      {!isUser && actions}
    </div>
  )
}
