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

export async function createSession(
  uploadId: string,
  provider?: ProviderConfig["provider"]
): Promise<SessionState> {
  return request("/session", {
    method: "POST",
    body: JSON.stringify({ upload_id: uploadId, provider }),
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

export async function getProviderConfig(opts?: {
  sessionId?: string;
  provider?: ProviderConfig["provider"];
}): Promise<ProviderConfig> {
  const params = new URLSearchParams();
  if (opts?.sessionId) params.set("session_id", opts.sessionId);
  if (opts?.provider) params.set("provider", opts.provider);
  const query = params.toString();
  return request(`/settings/provider${query ? `?${query}` : ""}`);
}

export async function switchProvider(
  provider: ProviderConfig["provider"],
  sessionId: string
): Promise<ProviderConfig> {
  return request("/settings/provider", {
    method: "POST",
    body: JSON.stringify({ provider, session_id: sessionId }),
  });
}

export async function getAvailableModels(opts?: {
  sessionId?: string;
  provider?: ProviderConfig["provider"];
}): Promise<ModelInfo[]> {
  const params = new URLSearchParams();
  if (opts?.sessionId) params.set("session_id", opts.sessionId);
  if (opts?.provider) params.set("provider", opts.provider);
  const query = params.toString();
  return request<{ models: ModelInfo[] }>(
    `/settings/models${query ? `?${query}` : ""}`
  ).then((r) => r.models);
}

// ── SSE Streaming ───────────────────────────────────────────────────────────

export function streamMessage(
  sessionId: string,
  content: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  onDone: () => void,
  onError: (error: Error) => void,
  provider?: ProviderConfig["provider"]
): AbortController {
  const controller = new AbortController();

  fetch(`${BASE}/session/${sessionId}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content, provider }),
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

      // Parse one SSE event block (lines separated by \n, terminated by a
      // blank line). sse-starlette emits standard SSE:
      //   event: <name>\n data: <json>\n\n
      const handleBlock = (block: string) => {
        let name = "message";
        const dataParts: string[] = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) {
            name = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataParts.push(line.slice(5).replace(/^ /, ""));
          }
        }
        if (dataParts.length === 0) return;
        try {
          const data = JSON.parse(dataParts.join("\n"));
          onEvent(name, data);
        } catch {
          // Skip malformed events
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Normalize CRLF (sse-starlette uses \r\n) so blank-line splitting works.
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        // Split off complete event blocks (delimited by a blank line),
        // keeping any trailing partial block in the buffer.
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          if (block.trim()) handleBlock(block);
        }
      }

      // Flush any remaining buffered event.
      if (buffer.trim()) handleBlock(buffer);

      onDone();
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError(err);
      }
    });

  return controller;
}
