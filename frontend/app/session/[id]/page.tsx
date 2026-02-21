"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, BookOpen, Moon, Sun, Settings } from "lucide-react";
import { useStore } from "@/lib/store";
import { useTheme } from "@/components/ThemeProvider";
import ChatInterface from "@/components/ChatInterface";
import SlideViewer from "@/components/SlideViewer";
import SlideNavigator from "@/components/SlideNavigator";
import ProgressBar from "@/components/ProgressBar";

export default function SessionPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const { theme, setTheme } = useTheme();
  const {
    session,
    slides,
    currentSlide,
    explanationMode,
    pacing,
    setExplanationMode,
    setPacing,
  } = useStore();

  // Redirect if no session
  useEffect(() => {
    if (!session) {
      router.push("/");
    }
  }, [session, router]);

  if (!session) return null;

  const activeSlide = slides.find((s) => s.slide_number === currentSlide) || null;

  return (
    <div className="flex h-screen flex-col bg-white dark:bg-gray-950">
      {/* Top bar */}
      <header className="flex items-center justify-between border-b border-gray-200 px-4 py-2 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <BookOpen className="h-5 w-5 text-brand-500" />
          <span className="font-semibold text-gray-900 dark:text-white">
            SlideGuide
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Explanation mode selector */}
          <select
            value={explanationMode}
            onChange={(e) => setExplanationMode(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="standard">Standard</option>
            <option value="analogy">Analogy</option>
            <option value="visual">Visual</option>
            <option value="step_by_step">Step-by-Step</option>
            <option value="eli5">ELI5</option>
          </select>

          {/* Pacing selector */}
          <select
            value={pacing}
            onChange={(e) => setPacing(e.target.value)}
            className="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="slow">Slow</option>
            <option value="medium">Medium</option>
            <option value="fast">Fast</option>
          </select>

          {/* Theme toggle */}
          <button
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-800"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </button>
        </div>
      </header>

      {/* Three-column layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Slide Viewer */}
        <div className="hidden w-80 flex-shrink-0 flex-col border-r border-gray-200 dark:border-gray-800 lg:flex">
          <div className="flex-1 overflow-hidden">
            <SlideViewer slide={activeSlide} />
          </div>
          <SlideNavigator />
        </div>

        {/* Center: Chat */}
        <div className="flex flex-1 flex-col">
          <ChatInterface />
        </div>

        {/* Right: Progress */}
        <div className="hidden w-72 flex-shrink-0 overflow-y-auto border-l border-gray-200 dark:border-gray-800 xl:block">
          <ProgressBar />
        </div>
      </div>
    </div>
  );
}
