import { cn } from "@/lib/utils"

function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="skeleton"
      className={cn("skeleton-pulse rounded-md bg-overlay-strong", className)}
      {...props}
    />
  )
}

export { Skeleton }
