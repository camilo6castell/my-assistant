import { Info } from "lucide-react"
import { cn } from "@/lib/utils"

/**
 * Ícono de información con tooltip on-hover, sin depender de un portal
 * (@base-ui/react no trae un Tooltip todavía en este proyecto -- ver
 * components.json). Suficiente para textos cortos de ayuda dentro de
 * paneles con overflow, como GenerationSection.
 */
export function InfoTooltip({ text, className }: { text: string; className?: string }) {
  return (
    <span className={cn("group/tooltip relative inline-flex", className)}>
      <Info className="size-3.5 text-muted-foreground/70 transition-colors hover:text-foreground" />
      <span
        role="tooltip"
        className={cn(
          "pointer-events-none absolute bottom-full left-1/2 z-20 mb-2 w-56 -translate-x-1/2",
          "rounded-lg border border-white/10 bg-popover px-2.5 py-2 text-[11px] leading-snug text-popover-foreground shadow-lg",
          "opacity-0 scale-95 transition-all duration-150",
          "group-hover/tooltip:opacity-100 group-hover/tooltip:scale-100"
        )}
      >
        {text}
      </span>
    </span>
  )
}
