"use client";

import { AlertTriangle, Cloud, Sparkles } from "lucide-react";
import { useStore } from "@/lib/store";
import { useEffect, useState } from "react";
import type { ProviderId } from "@/lib/types";

function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

export default function ProviderBanner() {
  const { provider, loadProviderConfig, switchProvider } = useStore();
  const [switching, setSwitching] = useState(false);
  const [switchError, setSwitchError] = useState<string | null>(null);

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

  const isCursor = provider.provider === "cursor";
  const isUnreachable = provider.endpoint?.status === "unreachable";
  const isUnconfigured = provider.endpoint?.status === "unconfigured";
  const host = isCursor ? "Cursor SDK" : hostOf(provider.base_url);
  const usageLabel = isCursor
    ? "billed to Cursor usage"
    : `OpenAI-compatible endpoint: ${host}`;

  const onSwitch = async (next: ProviderId) => {
    if (next === provider.provider || switching) return;
    setSwitching(true);
    setSwitchError(null);
    try {
      await switchProvider(next);
    } catch (err) {
      setSwitchError(err instanceof Error ? err.message : "Could not switch provider");
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div className="flex items-center gap-2 border-b border-gray-200 bg-gray-50 px-4 py-1.5 text-xs dark:border-gray-700 dark:bg-gray-900">
      {isUnreachable || isUnconfigured ? (
        <>
          <span className="h-2 w-2 rounded-full bg-red-500" />
          <AlertTriangle className="h-3.5 w-3.5 text-red-500" />
          <span className="text-red-600 dark:text-red-400">
            {isCursor
              ? isUnconfigured
                ? "Cursor SDK not configured (set CURSOR_API_KEY)"
                : "Cursor SDK unreachable"
              : `LLM endpoint unreachable (${host})`}
          </span>
        </>
      ) : (
        <>
          <span className="h-2 w-2 rounded-full bg-green-500" />
          {isCursor ? (
            <Sparkles className="h-3.5 w-3.5 text-brand-500" />
          ) : (
            <Cloud className="h-3.5 w-3.5 text-gray-500 dark:text-gray-400" />
          )}
          <span className="text-gray-600 dark:text-gray-300">
            {isCursor
              ? `Cursor SDK · ${provider.models.primary || "composer-2.5"} · ${usageLabel}`
              : usageLabel}
          </span>
          {provider.endpoint && provider.endpoint.models_loaded > 0 && (
            <span className="text-gray-400 dark:text-gray-500">
              ({provider.endpoint.models_loaded} model
              {provider.endpoint.models_loaded !== 1 ? "s" : ""} available)
            </span>
          )}
        </>
      )}

      {provider.available_providers?.length > 1 && (
        <label className="ml-2 flex items-center gap-1 text-gray-500 dark:text-gray-400">
          <span>SDK</span>
          <select
            value={provider.provider}
            disabled={switching}
            onChange={(e) => onSwitch(e.target.value as ProviderId)}
            className="rounded border border-gray-200 bg-white px-1.5 py-0.5 text-xs dark:border-gray-700 dark:bg-gray-900"
          >
            {provider.available_providers.map((option) => (
              <option key={option.id} value={option.id} disabled={!option.configured}>
                {option.label}
                {!option.configured ? " (not configured)" : ""}
              </option>
            ))}
          </select>
        </label>
      )}

      {switchError && (
        <span className="text-red-600 dark:text-red-400">{switchError}</span>
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
