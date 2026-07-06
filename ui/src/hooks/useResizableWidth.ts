import { useCallback, useRef } from "react"

interface UseResizableWidthOptions {
  width: number
  onChange: (width: number) => void
  min: number
  max: number
  /**
   * +1 si arrastrar el mouse hacia la DERECHA debe agrandar el panel
   * (caso: handle en el borde derecho de un panel a la izquierda).
   * -1 si arrastrar hacia la IZQUIERDA debe agrandarlo (caso: handle en
   * el borde izquierdo de un panel a la derecha).
   */
  growDirection: 1 | -1
}

/**
 * Devuelve un onPointerDown para poner en el elemento "handle". Usa
 * setPointerCapture para que el arrastre siga funcionando aunque el
 * cursor se salga del handle durante el drag (arrastres rápidos).
 */
export function useResizableWidth({
  width,
  onChange,
  min,
  max,
  growDirection,
}: UseResizableWidthOptions) {
  const startX = useRef(0)
  const startWidth = useRef(width)

  const onPointerDown = useCallback(
    (e: React.PointerEvent<HTMLElement>) => {
      e.preventDefault()
      startX.current = e.clientX
      startWidth.current = width

      const handle = e.currentTarget
      handle.setPointerCapture(e.pointerId)

      function handleMove(ev: PointerEvent) {
        const delta = (ev.clientX - startX.current) * growDirection
        const next = Math.min(max, Math.max(min, startWidth.current + delta))
        onChange(next)
      }

      function handleUp(ev: PointerEvent) {
        handle.releasePointerCapture(ev.pointerId)
        window.removeEventListener("pointermove", handleMove)
        window.removeEventListener("pointerup", handleUp)
      }

      window.addEventListener("pointermove", handleMove)
      window.addEventListener("pointerup", handleUp)
    },
    [width, onChange, min, max, growDirection]
  )

  return onPointerDown
}
