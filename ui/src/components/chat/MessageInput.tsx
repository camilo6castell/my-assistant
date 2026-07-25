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
      className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6 sm:pb-6"
      style={{ paddingBottom: "max(1rem, env(safe-area-inset-bottom))" }}
    >
      <div className="relative overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-shadow focus-within:shadow-md focus-within:border-primary/20">
        {/* Context toolbar -- subtle info strip above the textarea */}
        <div className="border-b border-border/50 px-3 py-2">
          <ChatToolbar conversation={conversation} />
        </div>

        {/* Textarea + send button */}
        <div className="flex items-end gap-2 p-3">
          <Textarea
            ref={textareaRef}
            value={value}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Type your question..."
            rows={1}
            className="max-h-[200px] flex-1 border-0 bg-transparent px-1 py-1 text-[15px] leading-relaxed sm:text-sm focus-visible:ring-0 focus-visible:ring-offset-0"
            disabled={disabled}
          />
          <Button
            size="icon"
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            aria-label="Send message"
            className="shrink-0 rounded-xl"
          >
            <ArrowUp className="size-4" strokeWidth={2.5} />
          </Button>
        </div>
      </div>
    </div>
  );
}
