"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";

export default function SlideNavigator() {
  const { slides, currentSlide, setCurrentSlide, sendMessage } = useStore();

  if (slides.length === 0) return null;

  const canGoPrev = currentSlide > 1;
  const canGoNext = currentSlide < slides.length;

  const goToSlide = (slideNum: number) => {
    setCurrentSlide(slideNum);
    sendMessage(`Let's look at slide ${slideNum}. Can you explain what's on it?`);
  };

  return (
    <div className="flex items-center justify-between border-t border-gray-200 px-4 py-2 dark:border-gray-800">
      <button
        onClick={() => canGoPrev && goToSlide(currentSlide - 1)}
        disabled={!canGoPrev}
        className={cn(
          "rounded-lg p-1.5 transition-colors",
          canGoPrev
            ? "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            : "text-gray-300 dark:text-gray-700"
        )}
      >
        <ChevronLeft className="h-5 w-5" />
      </button>

      {/* Slide picker */}
      <div className="flex items-center gap-1">
        <select
          value={currentSlide}
          onChange={(e) => goToSlide(Number(e.target.value))}
          className="rounded-md border border-gray-200 bg-white px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-900"
        >
          {slides.map((slide) => (
            <option key={slide.slide_number} value={slide.slide_number}>
              Slide {slide.slide_number}
              {slide.title ? ` — ${slide.title}` : ""}
            </option>
          ))}
        </select>
        <span className="text-xs text-gray-400">of {slides.length}</span>
      </div>

      <button
        onClick={() => canGoNext && goToSlide(currentSlide + 1)}
        disabled={!canGoNext}
        className={cn(
          "rounded-lg p-1.5 transition-colors",
          canGoNext
            ? "text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
            : "text-gray-300 dark:text-gray-700"
        )}
      >
        <ChevronRight className="h-5 w-5" />
      </button>
    </div>
  );
}
