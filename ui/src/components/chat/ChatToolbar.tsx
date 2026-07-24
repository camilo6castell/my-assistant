import type { Conversation } from "@/types/chat";
import { DEMO_MODE } from "@/lib/demo";
import { ConnectionStatus } from "@/components/layout/ConnectionStatus";

/**
 * Solo el contenido informativo (colecciones activas, web search, modo
 * demo) -- la tarjeta que lo envuelve visualmente vive en
 * MessageInput.tsx, apilada detrás del cuadro de texto. Se mantiene
 * como componente aparte porque la lógica de qué mensaje mostrar no
 * tiene nada que ver con el layout que lo contiene.
 */
export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  const hasContextSource =
    conversation.activeCollections.length > 0 || conversation.useWebSearch;

  let text: string;
  let title: string | undefined;

  if (DEMO_MODE) {
    text = "Demo mode -- talking directly to Gemini, no retrieval";
  } else if (hasContextSource) {
    const parts: string[] = [];
    if (conversation.activeCollections.length > 0) {
      parts.push(
        `${conversation.activeCollections.length} active collection(s)`,
      );
    }
    if (conversation.useWebSearch) parts.push("web search active");
    text = parts.join(" · ");
  } else {
    text = "No active collections -- answering directly, without RAG";
    title =
      "No active collections -- the model answers directly, without RAG (you can attach a one-off file in the right-hand panel)";
  }

  return (
    <div className="flex min-w-0 items-center justify-between gap-2 text-white/75">
      <span className="truncate text-xs" title={title}>
        {text}
      </span>
      <ConnectionStatus className="hidden shrink-0 sm:inline-flex" />
    </div>
  );
}
