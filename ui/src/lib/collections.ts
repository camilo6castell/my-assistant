/**
 * GET /collections devuelve una lista plana "namespace/coleccion"
 * (ver ContextManager.list_all() en el backend). Acá se agrupa por
 * namespace para el picker jerárquico del sidebar -- la selección real
 * sigue viviendo a nivel de hoja (conversation.activeCollections), esto
 * es solo la vista.
 */

export interface CollectionGroup {
  namespace: string
  items: string[] // nombres completos "namespace/hoja", ya ordenados
}

export function groupCollections(flat: string[]): CollectionGroup[] {
  const byNamespace = new Map<string, string[]>()

  for (const full of flat) {
    const slashIndex = full.indexOf("/")
    const namespace = slashIndex === -1 ? full : full.slice(0, slashIndex)
    const items = byNamespace.get(namespace) ?? []
    items.push(full)
    byNamespace.set(namespace, items)
  }

  return Array.from(byNamespace.entries())
    .map(([namespace, items]) => ({ namespace, items: [...items].sort() }))
    .sort((a, b) => a.namespace.localeCompare(b.namespace))
}

export type GroupSelectionState = "all" | "some" | "none"

export function groupSelectionState(
  group: CollectionGroup,
  active: string[]
): GroupSelectionState {
  const activeSet = new Set(active)
  const selectedCount = group.items.filter((item) => activeSet.has(item)).length
  if (selectedCount === 0) return "none"
  if (selectedCount === group.items.length) return "all"
  return "some"
}

/** "sociologia/Colombia" -> "Colombia" (nombre de hoja para mostrar). */
export function leafLabel(fullName: string): string {
  const slashIndex = fullName.indexOf("/")
  return slashIndex === -1 ? fullName : fullName.slice(slashIndex + 1)
}
