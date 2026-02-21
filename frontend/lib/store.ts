/**
 * Zustand store for SlideGuide session state.
 *
 * Manages: upload status, session state, message list,
 * streaming state, and slide data.
 */

import { create } from "zustand";
import type {
  ChatMessage,
  ProviderConfig,
  QuizScore,
  SessionState,
  SlideContent,
  UploadResponse,
} from "./types";
import * as api from "./api";

interface SlideGuideStore {
  // Upload state
  upload: UploadResponse | null;
  isUploading: boolean;
  uploadError: string | null;

  // Session state
  session: SessionState | null;
  isCreatingSession: boolean;

  // Messages
  messages: ChatMessage[];
  isStreaming: boolean;
  streamController: AbortController | null;

  // Slides
  slides: SlideContent[];
  currentSlide: number;

  // Progress
  topicsCovered: string[];
  quizScore: QuizScore;
  phase: string;

  // Settings
  explanationMode: string;
  pacing: string;

  // Provider
  provider: ProviderConfig | null;

  // Actions
  uploadFile: (file: File) => Promise<void>;
  startSession: (uploadId: string) => Promise<void>;
  sendMessage: (content: string) => void;
  stopStreaming: () => void;
  loadHistory: () => Promise<void>;
  loadSlides: () => Promise<void>;
  loadProviderConfig: () => Promise<void>;
  setCurrentSlide: (slide: number) => void;
  setExplanationMode: (mode: string) => void;
  setPacing: (pacing: string) => void;
  reset: () => void;
}

const initialState = {
  upload: null,
  isUploading: false,
  uploadError: null,
  session: null,
  isCreatingSession: false,
  messages: [],
  isStreaming: false,
  streamController: null,
  slides: [],
  currentSlide: 1,
  topicsCovered: [],
  quizScore: { correct: 0, total: 0 },
  phase: "greeting",
  explanationMode: "standard",
  pacing: "medium",
  provider: null,
};

export const useStore = create<SlideGuideStore>((set, get) => ({
  ...initialState,

  uploadFile: async (file: File) => {
    set({ isUploading: true, uploadError: null });
    try {
      const upload = await api.uploadFile(file);
      set({ upload, isUploading: false });

      // Poll for processing completion if needed
      if (upload.status === "PROCESSING") {
        const poll = setInterval(async () => {
          try {
            const updated = await api.getUpload(upload.id);
            if (updated.status !== "PROCESSING") {
              clearInterval(poll);
              set({ upload: updated });
            }
          } catch {
            clearInterval(poll);
          }
        }, 2000);
      }
    } catch (err) {
      set({
        isUploading: false,
        uploadError: err instanceof Error ? err.message : "Upload failed",
      });
    }
  },

  startSession: async (uploadId: string) => {
    set({ isCreatingSession: true });
    try {
      const session = await api.createSession(uploadId);
      set({
        session,
        isCreatingSession: false,
        phase: session.phase,
        topicsCovered: session.topics_covered,
        quizScore: (session.quiz_score as QuizScore) || { correct: 0, total: 0 },
      });

      // Load message history (greeting should be there)
      await get().loadHistory();
      await get().loadSlides();
    } catch (err) {
      set({ isCreatingSession: false });
      throw err;
    }
  },

  sendMessage: (content: string) => {
    const { session, isStreaming } = get();
    if (!session || isStreaming) return;

    // Add user message optimistically
    const userMsg: ChatMessage = {
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };

    // Add placeholder for assistant response
    const assistantMsg: ChatMessage = {
      role: "assistant",
      content: "",
      created_at: new Date().toISOString(),
    };

    set((state) => ({
      messages: [...state.messages, userMsg, assistantMsg],
      isStreaming: true,
    }));

    // Stream the response
    const controller = api.streamMessage(
      session.session_id,
      content,
      // onEvent
      (event, data) => {
        if (event === "token") {
          const text = (data as { text?: string }).text || "";
          set((state) => {
            const msgs = [...state.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, content: last.content + text };
            }
            return { messages: msgs };
          });
        } else if (event === "phase_change") {
          const phase = (data as { phase?: string }).phase || "";
          if (phase) set({ phase });
        } else if (event === "error") {
          const message = (data as { message?: string }).message || "An error occurred";
          set((state) => {
            const msgs = [...state.messages];
            const last = msgs[msgs.length - 1];
            if (last && last.role === "assistant") {
              msgs[msgs.length - 1] = { ...last, content: message };
            }
            return { messages: msgs, isStreaming: false };
          });
        }
      },
      // onDone
      () => {
        set({ isStreaming: false, streamController: null });
      },
      // onError
      (error) => {
        set((state) => {
          const msgs = [...state.messages];
          const last = msgs[msgs.length - 1];
          if (last && last.role === "assistant" && !last.content) {
            msgs[msgs.length - 1] = {
              ...last,
              content: "Sorry, something went wrong. Please try again.",
            };
          }
          return { messages: msgs, isStreaming: false, streamController: null };
        });
      }
    );

    set({ streamController: controller });
  },

  stopStreaming: () => {
    const { streamController } = get();
    if (streamController) {
      streamController.abort();
      set({ isStreaming: false, streamController: null });
    }
  },

  loadHistory: async () => {
    const { session } = get();
    if (!session) return;

    try {
      const history = await api.getHistory(session.session_id);
      set({ messages: history.messages });
    } catch {
      // Silent fail — messages will be empty
    }
  },

  loadSlides: async () => {
    const { upload } = get();
    if (!upload) return;

    try {
      const data = await api.getSlides(upload.id);
      set({ slides: data.slides });
    } catch {
      // Silent fail
    }
  },

  loadProviderConfig: async () => {
    try {
      const config = await api.getProviderConfig();
      set({ provider: config });
    } catch {
      // Silent fail — provider info is informational
    }
  },

  setCurrentSlide: (slide: number) => set({ currentSlide: slide }),
  setExplanationMode: (mode: string) => set({ explanationMode: mode }),
  setPacing: (pacing: string) => set({ pacing }),

  reset: () => {
    const { streamController } = get();
    if (streamController) streamController.abort();
    set(initialState);
  },
}));
