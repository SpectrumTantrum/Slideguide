"use client";

import { Cloud, AlertTriangle } from "lucide-react";
import { useStore } from "@/lib/store";
import { useEffect } from "react";

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export default function ProviderBanner() {
  const { provider, loadProviderConfig } = useStore();

  useEffect(() => {
    loadProviderConfig();
  }, [loadProviderConfig]);

  if (!provider) return null;

  const warnings: string[] = [];
  if (!provider.capabilities.vision) warnings.push("Vision unavailable");
  if (provider.capabilities.tool_mode === "prompt")
    warnings.push("Tool use: prompt-based");
  if (provider.capabilities.tool_mode === "none")
    warnings.push("Tool use unavailable");

  const isUnreachable = provider.endpoint?.status === "unreachable";
  const host = hostOf(provider.base_url);

  return (
    <div className="flex items-center gap-2 border-b border-gray-200 bg-gray-50 px-4 py-1.5 text-xs dark:border-gray-700 dark:bg-gray-900">
      {isUnreachable ? (
        <>
          <span className="h-2 w-2 rounded-full bg-red-500" />
          <AlertTriangle className="h-3.5 w-3.5 text-red-500" />
          <span className="text-red-600 dark:text-red-400">
            LLM endpoint unreachable ({host})
          </span>
        </>
      ) : (
        <>
          <span className="h-2 w-2 rounded-full bg-green-500" />
          <Cloud className="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" />
          <span className="text-gray-600 dark:text-gray-300">
            OpenAI-compatible endpoint: {host}
          </span>
          {provider.endpoint && provider.endpoint.models_loaded > 0 && (
            <span className="text-gray-400 dark:text-gray-500">
              ({provider.endpoint.models_loaded} model
              {provider.endpoint.models_loaded !== 1 ? "s" : ""} available)
            </span>
          )}
        </>
      )}

      {warnings.length > 0 && (
        <div className="ml-auto flex gap-1.5">
          {warnings.map((w) => (
            <span
              key={w}
              className="rounded bg-yellow-100 px-1.5 py-0.5 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300"
            >
              {w}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
