import { ArrowDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/types/chat";
import { MessageBubble } from "./MessageBubble";

/** Qué tan cerca del fondo (en px) se considera "está leyendo el final". */
const NEAR_BOTTOM_THRESHOLD = 120;

export function MessageList({
  messages,
  onDeleteMessage,
}: {
  messages: ChatMessage[]
  onDeleteMessage?: (messageId: string) => void
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Se lee de forma síncrona en el efecto de scroll -- no dispara
  // re-render por sí solo, a diferencia de guardarlo en estado.
  const isNearBottomRef = useRef(true);
  const [showJumpButton, setShowJumpButton] = useState(false);

  const lastMessage = messages[messages.length - 1];
  const lastMessageContent = lastMessage?.content;
  const lastMessageRole = lastMessage?.role;

  function handleScroll() {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const near = distanceFromBottom < NEAR_BOTTOM_THRESHOLD;
    isNearBottomRef.current = near;
    setShowJumpButton(!near);
  }

  // Auto-scroll "inteligente": si ya se estaba mirando el final de la
  // conversación (o el turno nuevo es del propio usuario, que siempre
  // espera ver su mensaje y la respuesta que sigue), baja solo. Si el
  // usuario había subido a releer algo más arriba, no lo interrumpe --
  // en cambio aparece el botón flotante para saltar al final cuando
  // quiera.
  useEffect(() => {
    if (!lastMessage) return;
    if (isNearBottomRef.current || lastMessageRole === "user") {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
      isNearBottomRef.current = true;
      setShowJumpButton(false);
    } else {
      setShowJumpButton(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, lastMessageContent, lastMessageRole]);

  function scrollToBottom() {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    isNearBottomRef.current = true;
    setShowJumpButton(false);
  }

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 overflow-y-auto px-4 py-6 sm:px-6 lg:px-8"
      >
        {messages.map((message) => (
          <MessageBubble
            key={message.id}
            message={message}
            onDelete={onDeleteMessage ? () => onDeleteMessage(message.id) : undefined}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {showJumpButton && (
        <button
          type="button"
          onClick={scrollToBottom}
          className="absolute bottom-4 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-border bg-overlay-strong px-3.5 py-2 text-xs font-medium text-foreground shadow-lg backdrop-blur-xl transition-transform duration-200 animate-in fade-in slide-in-from-bottom-2 hover:-translate-y-0.5 hover:bg-overlay-hover active:translate-y-0"
        >
          <ArrowDown className="size-3.5" />
          Jump to latest
        </button>
      )}
    </div>
  );
}
