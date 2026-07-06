import { Check, Minus } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  type GroupSelectionState,
  groupCollections,
  groupSelectionState,
  leafLabel,
} from "@/lib/collections"

function CheckboxIndicator({
  state,
  size = "size-4",
}: {
  state: GroupSelectionState
  size?: string
}) {
  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center rounded border transition-colors",
        size,
        state === "none" ? "border-white/20 bg-transparent" : "border-primary/50 bg-primary/20"
      )}
    >
      {state === "all" && <Check className="size-3 text-primary" strokeWidth={3} />}
      {state === "some" && <Minus className="size-3 text-primary" strokeWidth={3} />}
    </span>
  )
}

export function CollectionsPicker({
  collections,
  active,
  onChange,
  disabled,
}: {
  collections: string[]
  active: string[]
  onChange: (next: string[]) => void
  disabled?: boolean
}) {
  const groups = groupCollections(collections)

  function toggleGroup(items: string[], state: GroupSelectionState) {
    const next = new Set(active)
    if (state === "all") {
      for (const item of items) next.delete(item)
    } else {
      // "some" o "none" -> completar el grupo entero
      for (const item of items) next.add(item)
    }
    onChange(Array.from(next))
  }

  function toggleItem(item: string) {
    const next = new Set(active)
    if (next.has(item)) next.delete(item)
    else next.add(item)
    onChange(Array.from(next))
  }

  if (groups.length === 0) {
    return (
      <p className="px-2 py-6 text-center text-xs text-muted-foreground">
        No hay colecciones disponibles.
      </p>
    )
  }

  return (
    <div className="space-y-3">
      {groups.map((group) => {
        const state = groupSelectionState(group, active)
        return (
          <div key={group.namespace}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => toggleGroup(group.items, state)}
              className="group flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm text-foreground transition-colors hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <CheckboxIndicator state={state} />
              <span className="min-w-0 flex-1 truncate font-medium">{group.namespace}</span>
            </button>

            <div className="ml-[0.6875rem] space-y-0.5 border-l border-white/10 pl-2.5">
              {group.items.map((item) => {
                const isActive = active.includes(item)
                return (
                  <button
                    key={item}
                    type="button"
                    disabled={disabled}
                    onClick={() => toggleItem(item)}
                    className={cn(
                      "flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-xs transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                      isActive
                        ? "text-foreground"
                        : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                    )}
                  >
                    <CheckboxIndicator state={isActive ? "all" : "none"} size="size-3.5" />
                    <span className="min-w-0 flex-1 truncate">{leafLabel(item)}</span>
                  </button>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}
