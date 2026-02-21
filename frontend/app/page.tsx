"use client";

import { useRouter } from "next/navigation";
import { BookOpen, Brain, Sparkles, Zap } from "lucide-react";
import SlideUploader from "@/components/SlideUploader";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";

const FEATURES = [
  {
    icon: Brain,
    title: "Adaptive Learning",
    description: "Personalized explanations that adjust to your learning style and pace.",
  },
  {
    icon: Sparkles,
    title: "Multiple Modes",
    description: "Standard, analogy, visual, step-by-step, or ELI5 — learn your way.",
  },
  {
    icon: Zap,
    title: "Interactive Quizzes",
    description: "Test your knowledge with auto-generated questions that adapt to your level.",
  },
];

export default function HomePage() {
  const router = useRouter();
  const { upload, isCreatingSession, startSession } = useStore();

  const handleStartSession = async () => {
    if (!upload || upload.status !== "READY") return;
    try {
      await startSession(upload.id);
      router.push(`/session/${useStore.getState().session?.session_id}`);
    } catch (err) {
      console.error("Failed to start session:", err);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-gray-50 dark:from-gray-950 dark:to-gray-900">
      {/* Hero */}
      <div className="mx-auto max-w-4xl px-6 py-20 text-center">
        <div className="mb-6 flex items-center justify-center gap-2">
          <BookOpen className="h-10 w-10 text-brand-500" />
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 dark:text-white">
            SlideGuide
          </h1>
        </div>

        <p className="mx-auto max-w-2xl text-lg text-gray-600 dark:text-gray-300">
          Upload your lecture slides and get a personal AI tutor. Designed for
          how you actually learn — with patience, multiple explanations, and
          adaptive quizzes.
        </p>

        {/* Upload Zone */}
        <div className="mt-12">
          <SlideUploader />

          {upload && upload.status === "READY" && (
            <button
              onClick={handleStartSession}
              disabled={isCreatingSession}
              className={cn(
                "mt-6 inline-flex items-center gap-2 rounded-xl bg-brand-600 px-8 py-3 text-lg font-semibold text-white shadow-lg transition-all hover:bg-brand-700 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2",
                isCreatingSession && "cursor-not-allowed opacity-60"
              )}
            >
              {isCreatingSession ? (
                <>
                  <span className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Starting session...
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5" />
                  Start Tutoring Session
                </>
              )}
            </button>
          )}
        </div>

        {/* Features */}
        <div className="mt-20 grid gap-8 md:grid-cols-3">
          {FEATURES.map((feature) => (
            <div
              key={feature.title}
              className="rounded-xl border border-gray-200 bg-white p-6 text-left shadow-sm dark:border-gray-800 dark:bg-gray-900"
            >
              <feature.icon className="mb-3 h-8 w-8 text-brand-500" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {feature.title}
              </h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
