import type { Conversation } from "@/types/chat";
import { DEMO_MODE } from "@/lib/demo";
import { ConnectionStatus } from "@/components/layout/ConnectionStatus";
import { cn } from "@/lib/utils";
import {
  Globe,
  Layers,
  Search,
} from "lucide-react";

/**
 * Context toolbar rendered inside the input area. Shows active context
 * sources as subtle inline chips -- clean and informative without
 * competing with the textarea.
 */
export function ChatToolbar({ conversation }: { conversation: Conversation }) {
  const hasContextSource =
    conversation.activeCollections.length > 0 || conversation.useWebSearch;

  if (DEMO_MODE) {
    return (
      <div className="flex items-center justify-between gap-2 px-1">
        <span className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground/60">
          <span className="size-1.5 rounded-full bg-primary/60" />
          Demo mode
        </span>
        <ConnectionStatus className="hidden sm:inline-flex" />
      </div>
    );
  }

  if (!hasContextSource) {
    return (
      <div className="flex items-center justify-between gap-2 px-1">
        <span className="text-[11px] text-muted-foreground/50">
          No active collections
        </span>
        <ConnectionStatus className="hidden sm:inline-flex" />
      </div>
    );
  }

  return (
    <div className="flex items-center justify-between gap-2 px-1">
      <div className="flex min-w-0 items-center gap-1.5">
        {conversation.activeCollections.length > 0 && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
              "bg-primary/8 text-primary/80",
            )}
          >
            <Layers className="size-3" />
            {conversation.activeCollections.length}
          </span>
        )}
        {conversation.useWebSearch && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
              "bg-primary/8 text-primary/80",
            )}
          >
            <Globe className="size-3" />
            Web
          </span>
        )}
        {conversation.mode === "HARD" && (
          <span
            className={cn(
              "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium",
              "bg-primary/8 text-primary/80",
            )}
          >
            <Search className="size-3" />
            Strict
          </span>
        )}
      </div>
      <ConnectionStatus className="hidden shrink-0 sm:inline-flex" />
    </div>
  );
}
