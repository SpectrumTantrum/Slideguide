"use client";

import { useEffect, useRef, useState } from "react";
import {
  Send,
  Square,
  BookOpen,
  HelpCircle,
  ListChecks,
  RefreshCw,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";
import type { ChatMessage } from "@/lib/types";

const QUICK_ACTIONS = [
  { label: "Explain this", icon: BookOpen, message: "Can you explain this concept?" },
  { label: "Quiz me", icon: HelpCircle, message: "Quiz me on what we've covered." },
  { label: "Summarize", icon: ListChecks, message: "Give me a summary of what we've covered." },
  { label: "Different way", icon: RefreshCw, message: "I don't understand. Can you explain it differently?" },
];

export default function ChatInterface() {
  const { messages, isStreaming, sendMessage, stopStreaming } = useStore();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-focus input
  useEffect(() => {
    inputRef.current?.focus();
  }, [isStreaming]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    sendMessage(trimmed);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickAction = (message: string) => {
    if (isStreaming) return;
    sendMessage(message);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-gray-400 dark:text-gray-500">
            <p>Start by asking a question about your slides!</p>
          </div>
        )}

        <div className="space-y-4">
          {messages.map((msg, idx) => (
            <MessageBubble key={idx} message={msg} />
          ))}

          {/* Streaming indicator */}
          {isStreaming &&
            messages.length > 0 &&
            messages[messages.length - 1].role === "assistant" &&
            !messages[messages.length - 1].content && (
              <div className="flex items-center gap-2 px-4 py-2 text-sm text-gray-400">
                <span className="flex gap-1">
                  <span className="h-2 w-2 animate-pulse-dot rounded-full bg-brand-400" style={{ animationDelay: "0s" }} />
                  <span className="h-2 w-2 animate-pulse-dot rounded-full bg-brand-400" style={{ animationDelay: "0.2s" }} />
                  <span className="h-2 w-2 animate-pulse-dot rounded-full bg-brand-400" style={{ animationDelay: "0.4s" }} />
                </span>
                Thinking...
              </div>
            )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick actions */}
      {!isStreaming && messages.length > 0 && (
        <div className="flex gap-2 overflow-x-auto border-t border-gray-100 px-4 py-2 dark:border-gray-800">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.label}
              onClick={() => handleQuickAction(action.message)}
              className="flex flex-shrink-0 items-center gap-1.5 rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 transition-colors hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700 dark:border-gray-700 dark:text-gray-400 dark:hover:border-brand-700 dark:hover:bg-brand-950/30 dark:hover:text-brand-300"
            >
              <action.icon className="h-3.5 w-3.5" />
              {action.label}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 p-4 dark:border-gray-800">
        <div className="flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your slides..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm shadow-sm transition-colors placeholder:text-gray-400 focus:border-brand-400 focus:outline-none focus:ring-1 focus:ring-brand-400 dark:border-gray-700 dark:bg-gray-900 dark:placeholder:text-gray-500"
            style={{ maxHeight: "120px" }}
          />

          {isStreaming ? (
            <button
              onClick={stopStreaming}
              className="rounded-xl bg-red-500 p-3 text-white shadow-sm transition-colors hover:bg-red-600"
              title="Stop generating"
            >
              <Square className="h-5 w-5" />
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className={cn(
                "rounded-xl p-3 shadow-sm transition-colors",
                input.trim()
                  ? "bg-brand-600 text-white hover:bg-brand-700"
                  : "bg-gray-100 text-gray-400 dark:bg-gray-800 dark:text-gray-600"
              )}
              title="Send message"
            >
              <Send className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex animate-fade-in",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "max-w-[85%] rounded-2xl px-4 py-3 text-sm",
          isUser
            ? "bg-brand-600 text-white"
            : "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
        )}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:text-base prose-headings:font-semibold prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || "..."}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
