import { ArrowUp } from "lucide-react"
import { useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export function MessageInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void
  disabled?: boolean
}) {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSubmit() {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue("")
    if (textareaRef.current) textareaRef.current.style.height = "auto"
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setValue(e.target.value)
    const el = e.target
    el.style.height = "auto"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-6">
      <div className="flex items-end gap-2 rounded-2xl border border-border bg-overlay-strong p-2 shadow-2xl backdrop-blur-2xl">
        <Textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          placeholder="Type your question..."
          rows={1}
          className="max-h-[200px] flex-1 px-2 py-2"
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
  )
}
