import { useMutation, useQueryClient } from "@tanstack/react-query";
import { nanoid } from "nanoid";
import { Navigate, useParams } from "react-router-dom";
import {
  apiErrorMessage,
  isWebSearchQuotaExceededError,
  postQuery,
} from "@/lib/api/client";
import {
  askGeminiDemo,
  buildAttachmentsContext,
  DEMO_MODE,
  type DemoAnswer,
} from "@/lib/demo";
import { useConversationsStore } from "@/stores/conversationsStore";
import {
  useDemoAttachmentsStore,
  selectDemoFiles,
} from "@/stores/demoAttachmentsStore";
import type { ChatMessage } from "@/types/chat";
import type { QueryResponse } from "@/types/api";
import { MessageInput } from "./MessageInput";
import { MessageList } from "./MessageList";

function toHistory(
  messages: ChatMessage[],
): { user: string; assistant: string }[] {
  const history: { user: string; assistant: string }[] = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const a = messages[i];
    const b = messages[i + 1];
    if (
      a.role === "user" &&
      b.role === "assistant" &&
      !b.isPending &&
      !b.isError
    ) {
      history.push({ user: a.content, assistant: b.content });
    }
  }
  return history;
}

export function ChatView() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const conversation = useConversationsStore((s) =>
    s.conversations.find((c) => c.id === conversationId),
  );
  const addMessage = useConversationsStore((s) => s.addMessage);
  const updateMessage = useConversationsStore((s) => s.updateMessage);
  const deleteMessage = useConversationsStore((s) => s.deleteMessage);
  const setWebSearchQuotaExceeded = useConversationsStore(
    (s) => s.setWebSearchQuotaExceeded,
  );
  const queryClient = useQueryClient();
  const demoAttachments = useDemoAttachmentsStore(
    selectDemoFiles(conversation?.id ?? null),
  );
  const clearDemoAttachments = useDemoAttachmentsStore((s) => s.clearFiles);

  const sendMutation = useMutation<QueryResponse | DemoAnswer, unknown, string>(
    {
      mutationFn: (text: string) => {
        if (!conversation) throw new Error("Conversation not found");

        if (DEMO_MODE) {
          return askGeminiDemo({
            question: text,
            history: toHistory(conversation.messages),
            attachmentsContext: buildAttachmentsContext(demoAttachments),
          });
        }

        return postQuery(
          {
            question: text,
            collections: conversation.activeCollections,
            mode: conversation.mode,
            chat_history: toHistory(conversation.messages),
            conversation_id: conversation.id,
            generation: {
              max_tokens: conversation.generation.maxTokens,
              think_mode: conversation.generation.thinkMode,
            },
            web_search: conversation.useWebSearch,
          },
          conversation.useAgent,
        );
      },
    },
  );

  if (!conversationId || !conversation) {
    return <Navigate to="/" replace />;
  }

  function handleSend(text: string) {
    if (!conversation) return;
    const userMsg: ChatMessage = {
      id: nanoid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    const assistantId = nanoid();
    const pendingMsg: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      isPending: true,
      pendingLabel: !DEMO_MODE ? "compacting the response" : undefined,
    };

    addMessage(conversation.id, userMsg);
    addMessage(conversation.id, pendingMsg);

    sendMutation.mutate(text, {
      onSuccess: (data) => {
        if ("isDemo" in data) {
          updateMessage(conversation.id, assistantId, {
            content: data.answer,
            isPending: false,
          });
          clearDemoAttachments(conversation.id);
          return;
        }

        updateMessage(conversation.id, assistantId, {
          content: data.answer,
          confidence: data.confidence,
          collectionsUsed: data.collections_used,
          reformulated: data.reformulated,
          usedWebSearch: data.used_web_search,
          webSources: data.web_sources ?? undefined,
          isPending: false,
        });
        if (data.web_search_quota_exceeded) {
          setWebSearchQuotaExceeded(true);
        }
        queryClient.invalidateQueries({
          queryKey: ["attachments", conversation.id],
        });
      },
      onError: (error) => {
        updateMessage(conversation.id, assistantId, {
          content: apiErrorMessage(error),
          isPending: false,
          isError: true,
        });
        if (DEMO_MODE) return;
        if (isWebSearchQuotaExceededError(error)) {
          setWebSearchQuotaExceeded(true);
        }
      },
    });
  }

  function handleDeleteMessage(messageId: string) {
    if (!conversation) return;
    deleteMessage(conversation.id, messageId);
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative flex min-h-0 flex-1 flex-col">
        <MessageList
          messages={conversation.messages}
          onDeleteMessage={handleDeleteMessage}
        />
      </div>
      <MessageInput
        conversation={conversation}
        onSend={handleSend}
        disabled={sendMutation.isPending}
      />
    </div>
  );
}
