/** TypeScript interfaces mirroring the backend Pydantic schemas. */

export interface UploadResponse {
  id: string;
  filename: string;
  file_type: string;
  status: "PROCESSING" | "READY" | "ERROR";
  total_slides: number;
  created_at: string;
}

export interface SlideContent {
  slide_number: number;
  title: string | null;
  text_content: string;
  has_images: boolean;
}

export interface SlidesResponse {
  upload_id: string;
  total_slides: number;
  slides: SlideContent[];
}

export interface SessionState {
  session_id: string;
  upload_id: string;
  phase: string;
  current_slide: number | null;
  topics_covered: string[];
  quiz_score: Record<string, number>;
  message_count: number;
}

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tool_calls?: Record<string, unknown>[] | null;
  created_at?: string;
}

export interface MessageHistoryResponse {
  session_id: string;
  messages: ChatMessage[];
  total: number;
  limit: number;
  offset: number;
}

export interface RetrievalResult {
  content: string;
  slide_number: number;
  title: string;
  content_type: string;
  score: number;
  source: string;
}

export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface QuizScore {
  correct: number;
  total: number;
  confidence?: number;
}

export interface ProviderConfig {
  llm_provider: "openrouter" | "lmstudio";
  embedding_provider: "openai" | "lmstudio";
  vision_provider: "openrouter" | "lmstudio";
  capabilities: {
    vision: boolean;
    tool_mode: "native" | "prompt" | "none";
  };
  models: {
    primary: string;
    routing: string;
    embedding: string;
    vision: string;
  };
  lmstudio?: {
    status: "ok" | "unreachable";
    models_loaded: number;
  };
}

export interface ModelInfo {
  id: string;
  display_name?: string;
  supports_tools?: boolean;
  supports_vision?: boolean;
}
