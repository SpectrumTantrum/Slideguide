"use client";

import { useState } from "react";
import { CheckCircle2, XCircle, Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";

interface QuizCardProps {
  question: string;
  options?: string[];
  questionType?: "multiple_choice" | "short_answer" | "true_false";
}

export default function QuizCard({
  question,
  options,
  questionType = "multiple_choice",
}: QuizCardProps) {
  const { sendMessage } = useStore();
  const [selected, setSelected] = useState<string | null>(null);
  const [textAnswer, setTextAnswer] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    const answer = questionType === "short_answer" ? textAnswer : selected;
    if (!answer) return;

    setSubmitted(true);
    sendMessage(`My answer: ${answer}`);
  };

  if (questionType === "short_answer") {
    return (
      <div className="rounded-xl border border-brand-200 bg-brand-50/50 p-4 dark:border-brand-800 dark:bg-brand-950/20">
        <p className="mb-3 font-medium text-gray-900 dark:text-gray-100">
          {question}
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={textAnswer}
            onChange={(e) => setTextAnswer(e.target.value)}
            disabled={submitted}
            placeholder="Type your answer..."
            className="flex-1 rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm focus:border-brand-400 focus:outline-none dark:border-gray-700 dark:bg-gray-900"
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          <button
            onClick={handleSubmit}
            disabled={submitted || !textAnswer.trim()}
            className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  // Multiple choice or true/false
  const displayOptions =
    questionType === "true_false"
      ? ["True", "False"]
      : options || [];

  return (
    <div className="rounded-xl border border-brand-200 bg-brand-50/50 p-4 dark:border-brand-800 dark:bg-brand-950/20">
      <p className="mb-3 font-medium text-gray-900 dark:text-gray-100">
        {question}
      </p>

      <div className="space-y-2">
        {displayOptions.map((option, idx) => (
          <button
            key={idx}
            onClick={() => !submitted && setSelected(option)}
            disabled={submitted}
            className={cn(
              "flex w-full items-center gap-3 rounded-lg border px-4 py-2.5 text-left text-sm transition-colors",
              selected === option
                ? "border-brand-400 bg-brand-100 text-brand-900 dark:border-brand-600 dark:bg-brand-900/30 dark:text-brand-200"
                : "border-gray-200 bg-white hover:border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:hover:border-gray-600",
              submitted && "cursor-default"
            )}
          >
            <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border border-current text-xs font-medium">
              {String.fromCharCode(65 + idx)}
            </span>
            <span>{option}</span>
          </button>
        ))}
      </div>

      {!submitted && selected && (
        <button
          onClick={handleSubmit}
          className="mt-3 w-full rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700"
        >
          Submit Answer
        </button>
      )}

      {submitted && (
        <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">
          Answer submitted — waiting for feedback...
        </p>
      )}
    </div>
  );
}
