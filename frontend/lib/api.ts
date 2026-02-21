/**
 * Typed API client for the SlideGuide backend.
 *
 * All endpoints use the Next.js rewrite proxy (/api/* -> FastAPI),
 * so we don't need to specify the backend URL.
 */

import type {
  MessageHistoryResponse,
  ModelInfo,
  ProviderConfig,
  SessionState,
  SlidesResponse,
  UploadResponse,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || body.error || "Request failed");
  }

  return res.json();
}

// ── Upload ──────────────────────────────────────────────────────────────────

export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || "Upload failed");
  }

  return res.json();
}

export async function getUpload(uploadId: string): Promise<UploadResponse> {
  return request(`/upload/${uploadId}`);
}

export async function getSlides(uploadId: string): Promise<SlidesResponse> {
  return request(`/upload/${uploadId}/slides`);
}

// ── Session ─────────────────────────────────────────────────────────────────

export async function createSession(uploadId: string): Promise<SessionState> {
  return request("/session", {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId }),
  });
}

export async function getSession(sessionId: string): Promise<SessionState> {
  return request(`/session/${sessionId}`);
}

export async function getHistory(
  sessionId: string,
  limit = 50,
  offset = 0
): Promise<MessageHistoryResponse> {
  return request(`/session/${sessionId}/history?limit=${limit}&offset=${offset}`);
}

// ── Provider / Settings ─────────────────────────────────────────────────────

export async function getProviderConfig(): Promise<ProviderConfig> {
  return request("/settings/provider");
}

export async function getAvailableModels(): Promise<ModelInfo[]> {
  return request<{ models: ModelInfo[] }>("/settings/models").then(
    (r) => r.models
  );
}

// ── SSE Streaming ───────────────────────────────────────────────────────────

export function streamMessage(
  sessionId: string,
  content: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (error: Error) => void
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/session/${sessionId}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: "Stream failed" }));
        throw new ApiError(res.status, body.detail || "Message failed");
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              onEvent(payload.event, payload.data || payload);
            } catch {
              // Skip malformed events
            }
          }
          // Also handle sse-starlette format: event: X\ndata: Y
          if (line.startsWith("event: ")) {
            // Event name is on next data: line, handled above
          }
        }
      }

      onDone();
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError(err);
      }
    });

  return controller;
}
