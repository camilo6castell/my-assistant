import { useState } from "react";
import { nanoid } from "nanoid";
import { Navigate, useParams } from "react-router-dom";
import {
  apiErrorMessage,
  isWebSearchQuotaExceededError,
} from "@/lib/api/client";
import {
  DEMO_MODE,
  callDemoEndpointStream,
} from "@/lib/demo";
import {
  sendMessage,
  detectSendMode,
} from "@/lib/sendMessage";
import { useConversationsStore } from "@/stores/conversationsStore";
import {
  useDemoAttachmentsStore,
} from "@/stores/demoAttachmentsStore";
import type { ChatMessage } from "@/types/chat";
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
  const clearDemoAttachments = useDemoAttachmentsStore((s) => s.clearFiles);
  const [isSending, setIsSending] = useState(false);

  if (!conversationId || !conversation) {
    return <Navigate to="/" replace />;
  }

  function handleDeleteMessage(messageId: string) {
    if (!conversation) return;
    deleteMessage(conversation.id, messageId);
  }

  async function handleSend(text: string) {
    if (!conversation || isSending) return;
    setIsSending(true);
    const mode = detectSendMode();

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
      pendingLabel: mode === "demo_endpoint" ? "streaming response" : "compacting the response",
    };

    addMessage(conversation.id, userMsg);
    addMessage(conversation.id, pendingMsg);

    const baseParams = {
      question: text,
      collections: conversation.activeCollections,
      mode: conversation.mode,
      chatHistory: toHistory(conversation.messages),
      conversationId: conversation.id,
      generation: {
        maxTokens: conversation.generation.maxTokens,
        thinkMode: conversation.generation.thinkMode,
      },
      webSearch: conversation.useWebSearch,
    };

    if (mode === "demo_endpoint") {
      try {
        await callDemoEndpointStream(
          {
            question: text,
            collections: conversation.activeCollections,
            mode: conversation.mode,
          },
          (chunk) => {
            const msgs = useConversationsStore.getState().conversations
              .find((c) => c.id === conversation.id)?.messages ?? [];
            const current = msgs.find((m) => m.id === assistantId);
            updateMessage(conversation.id, assistantId, {
              content: (current?.content ?? "") + chunk,
            });
          },
          () => {
            updateMessage(conversation.id, assistantId, {
              isPending: false,
            });
          },
          (error) => {
            updateMessage(conversation.id, assistantId, {
              content: error,
              isPending: false,
              isError: true,
            });
          },
        );
      } catch (err) {
        updateMessage(conversation.id, assistantId, {
          content: apiErrorMessage(err),
          isPending: false,
          isError: true,
        });
      } finally {
        setIsSending(false);
      }
      return;
    }

    try {
      const result = await sendMessage(baseParams);
      if (result) {
        updateMessage(conversation.id, assistantId, {
          content: result.content,
          confidence: result.confidence,
          collectionsUsed: result.collectionsUsed,
          reformulated: result.reformulated,
          usedWebSearch: result.usedWebSearch,
          webSources: result.webSources,
          isPending: false,
        });
        if (result.webSearchQuotaExceeded) {
          setWebSearchQuotaExceeded(true);
        }
        if (mode === "demo") {
          clearDemoAttachments(conversation.id);
        }
      }
    } catch (err) {
      updateMessage(conversation.id, assistantId, {
        content: apiErrorMessage(err),
        isPending: false,
        isError: true,
      });
      if (!DEMO_MODE && isWebSearchQuotaExceededError(err)) {
        setWebSearchQuotaExceeded(true);
      }
    } finally {
      setIsSending(false);
    }
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
        disabled={isSending}
      />
    </div>
  );
}
