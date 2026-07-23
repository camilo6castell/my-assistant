import { cn } from "@/lib/utils"
import { useOnlineStatus } from "@/hooks/useOnlineStatus"

/**
 * Indicador de conectividad. En vez de un ícono permanente que ocupa
 * espacio incluso cuando todo está bien (ruido visual), solo se
 * expande a una etiqueta legible cuando hay algo que decir --
 * "Sin conexión". Online se reduce a un punto verde discreto, para no
 * competir con el resto de la toolbar.
 */
export function ConnectionStatus({ className }: { className?: string }) {
  const online = useOnlineStatus()

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 rounded-full text-[11px] font-medium transition-colors",
        online ? "text-muted-foreground" : "bg-destructive/10 px-2 py-1 text-destructive",
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
