import { ArrowUp } from "lucide-react";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { Conversation } from "@/types/chat";
import { ChatToolbar } from "./ChatToolbar";

export function MessageInput({
  conversation,
  onSend,
  disabled,
}: {
  conversation: Conversation;
  onSend: (text: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function handleSubmit() {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  return (
    <div
      className="mx-auto w-full max-w-3xl px-3 pb-4 sm:px-4 sm:pb-6"
      style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}
    >
      <div className="relative">
        {/* Tarjeta de contexto: mismo tamaño/forma que el cuadro de texto de
            abajo, pero apilada detrás -- un glass oscuro que no depende del
            tema (siempre oscuro, como un chip flotante) para distinguirse
            claramente como una capa "detrás" del input. El padding inferior
            extra (pb-9) es a propósito: es lo que el input de abajo tapa. */}
        <div className="relative z-0 rounded-2xl border border-white/10 bg-neutral-950/75 px-4 pt-3 pb-9 shadow-xl backdrop-blur-2xl">
          <ChatToolbar conversation={conversation} />
        </div>

        {/* Cuadro de texto: se monta -mt-7 sobre la tarjeta de arriba,
            tapando su mitad inferior. Lo que sobra por encima (el segmento
            con la info de colecciones / estado de conexión) queda visible
            como una pestaña que asoma detrás. */}
        <div className="relative z-10 -mt-7 flex items-end gap-2 rounded-2xl border border-border bg-overlay-strong p-2 shadow-2xl backdrop-blur-2xl">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Type your question..."
            rows={1}
            className="max-h-[200px] flex-1 px-2 py-2 text-base sm:text-sm"
            disabled={disabled}
          />
          <Button
            size="icon"
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            aria-label="Send message"
          >
            <ArrowUp className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
