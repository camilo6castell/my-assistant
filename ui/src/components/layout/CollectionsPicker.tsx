import { Check, ChevronRight, Minus } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  type GroupSelectionState,
  groupCollections,
  groupSelectionState,
  leafLabel,
} from "@/lib/collections";

function CheckboxIndicator({
  state,
  size = "size-4",
}: {
  state: GroupSelectionState;
  size?: string;
}) {
  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center rounded border transition-colors",
        size,
        state === "none"
          ? "border-border bg-transparent"
          : "border-primary/50 bg-primary/20",
      )}
    >
      {state === "all" && (
        <Check className="size-3 text-primary" strokeWidth={3} />
      )}
      {state === "some" && (
        <Minus className="size-3 text-primary" strokeWidth={3} />
      )}
    </span>
  );
}

export function CollectionsPicker({
  collections,
  active,
  onChange,
  disabled,
}: {
  collections: string[];
  active: string[];
  onChange: (next: string[]) => void;
  disabled?: boolean;
}) {
  const groups = groupCollections(collections);

  // Abiertos por defecto: los grupos que ya tienen algo seleccionado.
  // El inicializador de useState (la función de abajo) solo se evalúa
  // una vez, en el primer render -- por eso alcanza para lograr el
  // "calcular una sola vez al montar" que antes se intentaba armar a
  // mano con useMemo + setState (lo cual React desaconseja: llamar
  // setState dentro de useMemo puede disparar un loop infinito, porque
  // cada llamada cambia el estado que a su vez puede volver a evaluar
  // el memo). Ver https://react.dev/reference/react/useState#avoiding-recreating-the-initial-state
  const [openGroups, setOpenGroups] = useState<Set<string>>(
    () =>
      new Set(
        groups
          .filter((g) => groupSelectionState(g, active) !== "none")
          .map((g) => g.namespace),
      ),
  );

  function toggleOpen(namespace: string) {
    setOpenGroups((prev) => {
      const next = new Set(prev);
      if (next.has(namespace)) next.delete(namespace);
      else next.add(namespace);
      return next;
    });
  }

  function toggleGroup(items: string[], state: GroupSelectionState) {
    const next = new Set(active);
    if (state === "all") {
      for (const item of items) next.delete(item);
    } else {
      // "some" o "none" -> completar el grupo entero
      for (const item of items) next.add(item);
    }
    onChange(Array.from(next));
  }

  function toggleItem(item: string) {
    const next = new Set(active);
    if (next.has(item)) next.delete(item);
    else next.add(item);
    onChange(Array.from(next));
  }

  if (groups.length === 0) {
    return (
      <p className="px-2 py-6 text-center text-xs text-muted-foreground">
        No collections available.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      {groups.map((group) => {
        const state = groupSelectionState(group, active);
        const isOpen = openGroups.has(group.namespace);
        const selectedCount = group.items.filter((item) =>
          active.includes(item),
        ).length;

        return (
          <div
            key={group.namespace}
            className="overflow-hidden rounded-md border border-border"
          >
            {/* Header: chevron (expandir/colapsar) + checkbox (seleccionar grupo) */}
            <div className="flex items-center gap-1 bg-overlay pr-2">
              <button
                type="button"
                disabled={disabled}
                onClick={() => toggleOpen(group.namespace)}
                className="flex size-8 shrink-0 items-center justify-center text-muted-foreground transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ChevronRight
                  className={cn(
                    "size-3.5 transition-transform duration-200",
                    isOpen && "rotate-90",
                  )}
                />
              </button>

              <button
                type="button"
                disabled={disabled}
                onClick={() => toggleGroup(group.items, state)}
                className="group flex min-w-0 flex-1 items-center gap-2 rounded-md py-1.5 text-left text-sm text-foreground transition-colors hover:bg-overlay-hover disabled:cursor-not-allowed disabled:opacity-40"
              >
                <CheckboxIndicator state={state} />
                <span className="min-w-0 flex-1 truncate font-medium">
                  {group.namespace}
                </span>
              </button>

              {selectedCount > 0 && (
                <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                  {selectedCount}/{group.items.length}
                </span>
              )}
            </div>

            {/* Cuerpo colapsable, altura animada con grid-template-rows */}
            <div
              className={cn(
                "grid transition-[grid-template-rows] duration-200 ease-in-out",
                isOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
              )}
            >
              <div className="overflow-hidden">
                <div className="ml-4 flex flex-col gap-0.5 border-l border-border py-1 pl-2.5 pr-2">
                  {group.items.map((item) => {
                    const isActive = active.includes(item);
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
                            : "text-muted-foreground hover:bg-overlay-hover hover:text-foreground",
                        )}
                      >
                        <CheckboxIndicator
                          state={isActive ? "all" : "none"}
                          size="size-3.5"
                        />
                        <span className="min-w-0 flex-1 truncate">
                          {leafLabel(item)}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
