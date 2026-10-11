// Thin client for the CineQuery backend (FastAPI). All calls go straight from the browser.

export const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

// ---------------------------------------------------------------- types (mirror app/api/schemas)
export type JobStatus = "new" | "queued" | "running" | "done" | "failed";

export interface VideoSummary {
  video_id: string;
  title: string;
  status: JobStatus;
  duration_ms: number | null;
  created_at: string | null;
  overall: number;
}

export interface Progress {
  video_id: string;
  status: JobStatus;
  stage: string | null;
  fraction: number;
  message: string;
  overall: number;
  queue_position: number | null;
  stages: Record<string, string>;
  error?: string | null;
}

export interface VideoDetail {
  video_id: string;
  title: string;
  progress: Progress;
  ready: boolean;
  media: { duration_ms?: number; width?: number; height?: number; fps?: number };
  summary: Record<string, unknown>;
}

export interface Scene {
  scene: number;
  start_ms: number;
  end_ms: number;
  title: string;
  description: string;
  characters: string[];
  location: string;
  time_of_day: string;
  mood: string;
  events: string[];
  graphic: boolean;
  tags: Record<string, number>;
  tag_evidence?: Record<string, string>;
}

export interface Chapter {
  chapter: number;
  start_ms: number;
  end_ms: number;
  title?: string;
  summary?: string;
  events?: string[];
  characters?: string[];
  error?: string;
}

export interface Profile {
  video_type?: string;
  genres?: { genre: string; confidence: number; evidence: string }[];
  logline?: string;
  plot?: string;
  ending?: string;
  setting?: string;
  tone?: string;
  themes?: string[];
  characters?: { name: string; role: string; description: string }[];
  tag_stats?: Record<string, number>;
  error?: string;
}

export interface Source {
  label: string;
  start_ms: number;
  end_ms: number;
}

export interface ToolCall {
  action: string;
  args: Record<string, unknown>;
  source: string;
  seconds: number;
}

export interface AskResult {
  answer: string;
  sources: Source[];
  unverified_times: string[];
  model: string;
  seconds: number;
  tool_calls: ToolCall[];
}

// ---------------------------------------------------------------- plain requests
async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

export const listVideos = () => getJSON<VideoSummary[]>("/api/videos");
export const getVideo = (id: string) => getJSON<VideoDetail>(`/api/videos/${id}`);
export const getScenes = (id: string) => getJSON<Scene[]>(`/api/videos/${id}/scenes`);
export const getChapters = (id: string) => getJSON<Chapter[]>(`/api/videos/${id}/chapters`);
export const getProfile = (id: string) => getJSON<Profile>(`/api/videos/${id}/profile`);
export const streamUrl = (id: string) => `${API_URL}/api/videos/${id}/stream`;

export async function restartProcessing(id: string): Promise<Progress> {
  const res = await fetch(`${API_URL}/api/videos/${id}/process`, { method: "POST" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function importLocal(path: string): Promise<{ video_id: string }> {
  const res = await fetch(`${API_URL}/api/videos/import`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

/** Upload with a progress callback (XHR, because fetch has no upload progress). */
export function uploadVideo(
  file: File,
  onProgress: (fraction: number) => void,
): { promise: Promise<{ video_id: string }>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ video_id: string }>((resolve, reject) => {
    xhr.open("POST", `${API_URL}/api/videos?filename=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else {
        let msg = `${xhr.status} ${xhr.statusText}`;
        try { msg = JSON.parse(xhr.responseText).detail ?? msg; } catch { /* keep status text */ }
        reject(new Error(msg));
      }
    };
    xhr.onerror = () => reject(new Error("Upload failed: is the backend running on " + API_URL + "?"));
    xhr.onabort = () => reject(new Error("Upload cancelled"));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

// ---------------------------------------------------------------- live streams (Server-Sent Events)
/** Live processing progress. Returns a function that closes the stream. */
export function watchProgress(id: string, onUpdate: (p: Progress) => void, onError?: () => void): () => void {
  const es = new EventSource(`${API_URL}/api/videos/${id}/events`);
  const handle = (e: MessageEvent) => onUpdate(JSON.parse(e.data));
  es.addEventListener("progress", handle);
  es.addEventListener("done", (e) => { handle(e as MessageEvent); es.close(); });
  es.addEventListener("failed", (e) => { handle(e as MessageEvent); es.close(); });
  es.onerror = () => onError?.();
  return () => es.close();
}

export interface AskHandlers {
  onStatus?: (message: string) => void;
  onStep?: (step: ToolCall) => void;
  onToken?: (text: string) => void;
  onReset?: () => void;
  onAnswer?: (result: AskResult) => void;
  onError?: (message: string) => void;
}

/** Ask a question; events arrive as the backend works. (EventSource can't POST, so fetch + a small SSE parser.) */
export async function askStream(id: string, question: string, h: AskHandlers, signal?: AbortSignal): Promise<void> {
  const res = await fetch(`${API_URL}/api/videos/${id}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorText(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue;                       // ping / comment
      const payload = JSON.parse(data.join("\n"));
      if (event === "status") h.onStatus?.(payload.message);
      else if (event === "step") h.onStep?.(payload);
      else if (event === "token") h.onToken?.(payload.text);
      else if (event === "reset") h.onReset?.();
      else if (event === "answer") h.onAnswer?.(payload);
      else if (event === "error") h.onError?.(payload.message);
    }
  }
}
