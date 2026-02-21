"use client";

import { BarChart3, Target, BookCheck, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";

const PHASE_LABELS: Record<string, string> = {
  greeting: "Getting Started",
  topic_selection: "Choosing Topic",
  teaching: "Learning",
  quiz: "Quiz Time",
  review: "Reviewing",
  wrap_up: "Session Complete",
};

export default function ProgressBar() {
  const { topicsCovered, quizScore, phase, slides, explanationMode, pacing } =
    useStore();

  const totalSlides = slides.length;
  const coveragePercent = totalSlides > 0
    ? Math.round((topicsCovered.length / totalSlides) * 100)
    : 0;

  const quizAccuracy =
    quizScore.total > 0
      ? Math.round((quizScore.correct / quizScore.total) * 100)
      : 0;

  return (
    <div className="space-y-4 p-4">
      {/* Phase indicator */}
      <div className="rounded-lg bg-brand-50 p-3 dark:bg-brand-950/30">
        <span className="text-xs font-medium uppercase tracking-wider text-brand-600 dark:text-brand-400">
          Current Phase
        </span>
        <p className="mt-0.5 font-semibold text-gray-900 dark:text-white">
          {PHASE_LABELS[phase] || phase}
        </p>
      </div>

      {/* Coverage bar */}
      <div>
        <div className="mb-1 flex items-center justify-between text-xs">
          <span className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
            <BookCheck className="h-3.5 w-3.5" />
            Topics Covered
          </span>
          <span className="font-medium text-gray-900 dark:text-white">
            {topicsCovered.length}
            {totalSlides > 0 && ` / ${totalSlides}`}
          </span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
          <div
            className="h-full rounded-full bg-brand-500 transition-all duration-500"
            style={{ width: `${Math.min(coveragePercent, 100)}%` }}
          />
        </div>
      </div>

      {/* Quiz score */}
      {quizScore.total > 0 && (
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
              <Target className="h-3.5 w-3.5" />
              Quiz Score
            </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {quizScore.correct}/{quizScore.total} ({quizAccuracy}%)
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                quizAccuracy >= 70
                  ? "bg-green-500"
                  : quizAccuracy >= 40
                    ? "bg-yellow-500"
                    : "bg-red-500"
              )}
              style={{ width: `${quizAccuracy}%` }}
            />
          </div>
        </div>
      )}

      {/* Confidence meter */}
      {quizScore.confidence !== undefined && quizScore.confidence > 0 && (
        <div>
          <div className="mb-1 flex items-center justify-between text-xs">
            <span className="flex items-center gap-1 text-gray-600 dark:text-gray-400">
              <Gauge className="h-3.5 w-3.5" />
              Confidence
            </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {Math.round((quizScore.confidence as number) * 100)}%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
            <div
              className="h-full rounded-full bg-purple-500 transition-all duration-500"
              style={{ width: `${(quizScore.confidence as number) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Settings display */}
      <div className="mt-4 space-y-2 border-t border-gray-100 pt-4 dark:border-gray-800">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Mode</span>
          <span className="rounded-md bg-gray-100 px-2 py-0.5 font-medium capitalize text-gray-700 dark:bg-gray-800 dark:text-gray-300">
            {explanationMode}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Pacing</span>
          <span className="rounded-md bg-gray-100 px-2 py-0.5 font-medium capitalize text-gray-700 dark:bg-gray-800 dark:text-gray-300">
            {pacing}
          </span>
        </div>
      </div>

      {/* Topics list */}
      {topicsCovered.length > 0 && (
        <div className="border-t border-gray-100 pt-4 dark:border-gray-800">
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-gray-500">
            Topics Covered
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {topicsCovered.map((topic, idx) => (
              <span
                key={idx}
                className="rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900/30 dark:text-green-300"
              >
                {topic}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
