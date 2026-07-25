import { MessageSquarePlus, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

export function SidebarSectionHeader({
  icon: Icon,
  label,
  count,
  actionHandler,
}: {
  icon: LucideIcon;
  label: string;
  count?: number | string;
  actionHandler?: () => void;
}) {
  return (
    <div className="flex items-center gap-1.5 px-4 pb-2.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground/70">
      <Icon className="size-3.5" />
      <span>{label}</span>
      {count !== undefined && (
        <span className="text-muted-foreground/40 tabular-nums">{count}</span>
      )}
      {actionHandler && (
        <div className="ml-auto">
          <Button
            variant="ghost"
            size="xs"
            className="gap-1.5 text-muted-foreground hover:text-foreground"
            onClick={actionHandler}
          >
            <MessageSquarePlus className="size-3.5" />
            New chat
          </Button>
        </div>
      )}
    </div>
  );
}
