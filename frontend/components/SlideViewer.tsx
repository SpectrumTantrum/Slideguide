"use client";

import { FileText, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SlideContent } from "@/lib/types";

interface SlideViewerProps {
  slide: SlideContent | null;
}

export default function SlideViewer({ slide }: SlideViewerProps) {
  if (!slide) {
    return (
      <div className="flex h-full items-center justify-center text-gray-400 dark:text-gray-500">
        <div className="text-center">
          <FileText className="mx-auto mb-2 h-10 w-10" />
          <p className="text-sm">No slide selected</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4 scrollbar-thin">
      {/* Slide header */}
      <div className="mb-4 flex items-center justify-between">
        <div>
          <span className="text-xs font-medium uppercase tracking-wider text-brand-600 dark:text-brand-400">
            Slide {slide.slide_number}
          </span>
          {slide.title && (
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
              {slide.title}
            </h2>
          )}
        </div>
        {slide.has_images && (
          <span className="flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">
            <ImageIcon className="h-3 w-3" />
            Images
          </span>
        )}
      </div>

      {/* Slide content */}
      <div className="prose prose-sm max-w-none flex-1 dark:prose-invert">
        {slide.text_content ? (
          <div className="whitespace-pre-wrap rounded-lg border border-gray-100 bg-gray-50 p-4 text-sm leading-relaxed dark:border-gray-800 dark:bg-gray-900">
            {slide.text_content}
          </div>
        ) : (
          <p className="text-gray-400 italic">
            This slide has no text content.
            {slide.has_images && " It may contain images or diagrams."}
          </p>
        )}
      </div>
    </div>
  );
}
