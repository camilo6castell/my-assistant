import type { LucideIcon } from "lucide-react";

export function SidebarSectionHeader({
  icon: Icon,
  label,
  count,
}: {
  icon: LucideIcon;
  label: string;
  count?: number | string;
}) {
  return (
    <div className="flex items-center gap-1.5 px-3 pb-2 text-xs font-medium text-muted-foreground">
      <Icon className="size-3.5" />
      <span>{label}</span>
      {count !== undefined && (
        <span className="text-muted-foreground/50">{count}</span>
      )}
    </div>
  );
}
