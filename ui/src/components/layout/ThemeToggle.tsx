import { Monitor, Moon, Sun } from "lucide-react"
import { cn } from "@/lib/utils"
import { useUiStore, type Theme } from "@/stores/uiStore"

const OPTIONS: { value: Theme; icon: typeof Sun; label: string }[] = [
  { value: "light", icon: Sun, label: "Light theme" },
  { value: "system", icon: Monitor, label: "Match system theme" },
  { value: "dark", icon: Moon, label: "Dark theme" },
]

export function ThemeToggle({ className }: { className?: string }) {
  const theme = useUiStore((s) => s.theme)
  const setTheme = useUiStore((s) => s.setTheme)

  return (
    <div
      role="radiogroup"
      aria-label="Theme"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-lg border border-border bg-overlay p-0.5",
        className
      )}
    >
      {OPTIONS.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={theme === value}
          title={label}
          onClick={() => setTheme(value)}
          className={cn(
            "flex size-7 items-center justify-center rounded-md transition-all duration-150",
            theme === value
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Icon className="size-3.5" />
        </button>
      ))}
    </div>
  )
}
