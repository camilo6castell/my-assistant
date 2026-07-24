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
    <div className="flex items-center gap-1.5 px-3 pb-2 text-xs font-medium text-muted-foreground">
      <Icon className="size-3.5" />
      <span>{label}</span>
      {count !== undefined && (
        <span className="text-muted-foreground/50">{count}</span>
      )}
      {actionHandler && (
        <div className="ml-auto">
          <Button
            variant="secondary"
            className="w-full justify-start gap-1.5 bg-overlay-strong hover:bg-overlay-hover"
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
