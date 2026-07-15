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
            className="w-full justify-start gap-2 bg-white/[0.06] hover:bg-white/[0.1]"
            onClick={actionHandler}
          >
            <MessageSquarePlus className="size-4" />
            New chat
          </Button>
        </div>
      )}
    </div>
  );
}
