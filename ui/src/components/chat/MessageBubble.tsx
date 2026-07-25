import {
  AlertCircle,
  Check,
  Copy,
  Globe,
  Layers,
  ChevronDown,
  Sparkles,
  Trash2,
} from "lucide-react";
import { useState, type ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { ThinkingOrb } from "thinking-orbs";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types/chat";

/**
 * Shared copy button -- icon rotates to check for 1.5s as confirmation,
 * no external toast dependency.
 */
function CopyButton({
  getText,
  className,
  label = "Copy",
}: {
  getText: () => string;
  className?: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(getText());
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API can fail (permissions, insecure context)
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      aria-label={copied ? "Copied" : label}
      title={copied ? "Copied!" : label}
      className={cn(
        "inline-flex items-center justify-center rounded-lg p-1.5 text-muted-foreground transition-all hover:bg-overlay-hover hover:text-foreground",
        className,
      )}
    >
      {copied ? (
        <Check className="size-3.5 text-emerald-500" />
      ) : (
        <Copy className="size-3.5" />
      )}
    </button>
  );
}

/**
 * Custom <pre> with copy button -- always visible on mobile, hover-only on desktop.
 */
function CodeBlock({ children, className, ...props }: ComponentProps<"pre">) {
  return (
    <div className="group/code relative">
      <CopyButton
        getText={() => extractText(children)}
        label="Copy code"
        className="absolute right-2 top-2 z-10 bg-overlay-strong opacity-70 backdrop-blur-sm lg:opacity-0 lg:group-hover/code:opacity-100 transition-opacity"
      />
      <pre className={className} {...props}>
        {children}
      </pre>
    </div>
  );
}

/** Extract plain text from React children for clipboard copy. */
function extractText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object" && "props" in node) {
    return extractText(
      (node as { props: { children?: React.ReactNode } }).props.children,
    );
  }
  return "";
}

/**
 * Metadata footer for assistant messages. Collapsible to reduce visual
 * noise -- expanded by default for the latest message, collapsed for
 * older ones.
 */
function MessageMetadata({ message }: { message: ChatMessage }) {
  const [expanded, setExpanded] = useState(false);

  const hasMetadata =
    message.confidence !== undefined ||
    !!message.collectionsUsed?.length ||
    message.usedWebSearch ||
    message.reformulated;

  if (!hasMetadata) return null;

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground/50 hover:text-muted-foreground transition-colors"
      >
        <ChevronDown
          className={cn(
            "size-3 transition-transform duration-150",
            expanded && "rotate-180",
          )}
        />
        <span>Details</span>
      </button>
      {expanded && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground/60">
          {message.confidence !== undefined && (
            <span className="inline-flex items-center gap-1">
              <Sparkles className="size-3" />
              {(message.confidence * 100).toFixed(0)}% confidence
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
              web search
            </span>
          )}
          {message.reformulated && <span>reformulated</span>}
        </div>
      )}
    </div>
  );
}

export function MessageBubble({
  message,
  onDelete,
}: {
  message: ChatMessage;
  onDelete?: () => void;
}) {
  const isUser = message.role === "user";
  const showActions = !message.isPending && !message.isError;

  const actions = (
    <span
      className={cn(
        "mt-1 flex shrink-0 items-center gap-0.5 self-start opacity-0 transition-opacity duration-150 group-hover:opacity-100",
        "max-lg:opacity-100",
      )}
    >
      {showActions && (
        <CopyButton getText={() => message.content} label="Copy message" />
      )}
      {onDelete && showActions && (
        <button
          type="button"
          onClick={onDelete}
          aria-label="Delete message"
          title="Delete message"
          className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-overlay-hover hover:text-destructive"
        >
          <Trash2 className="size-3.5" />
        </button>
      )}
    </span>
  );

  return (
    <div
      className={cn(
        "group flex w-full items-start gap-2 message-enter",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {isUser && actions}
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed sm:max-w-[75ch]",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md"
            : "bg-transparent text-foreground",
          message.isError &&
            "border border-destructive/20 bg-destructive/5 text-destructive",
        )}
      >
        {message.isPending ? (
          <span className="inline-flex items-center gap-5">
            <ThinkingOrb state="composing" size={64} speed={1.25} />
            {message.pendingLabel && (
              <span className="text-xs text-muted-foreground">
                {message.pendingLabel}
              </span>
            )}
          </span>
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
          <MessageMetadata message={message} />
        )}

        {!!message.webSources?.length && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {message.webSources.map((source) => (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                title={source.url}
                className="inline-flex max-w-[220px] items-center gap-1 truncate rounded-lg border border-border bg-overlay px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-overlay-hover hover:text-foreground"
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
  );
}
