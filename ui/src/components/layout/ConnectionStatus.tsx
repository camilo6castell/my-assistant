import { cn } from "@/lib/utils"
import { useOnlineStatus } from "@/hooks/useOnlineStatus"

export function ConnectionStatus({ className }: { className?: string }) {
  const online = useOnlineStatus()

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full text-[11px] font-medium transition-colors",
        online ? "text-muted-foreground/50" : "bg-destructive/10 px-2 py-1 text-destructive",
        className
      )}
      role="status"
      aria-live="polite"
    >
      <span
        className={cn(
          "size-1.5 shrink-0 rounded-full",
          online ? "bg-emerald-500" : "animate-pulse bg-destructive"
        )}
        aria-hidden
      />
      {!online && "No internet connection"}
    </span>
  )
}
