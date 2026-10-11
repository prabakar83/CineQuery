// Movie time helpers (milliseconds everywhere, like the backend).

export function fmt(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = h ? String(m).padStart(2, "0") : String(m);
  return (h ? `${h}:` : "") + `${mm}:${String(sec).padStart(2, "0")}`;
}

export function parseTime(text: string): number | null {
  const parts = text.split(":").map(Number);
  if (!parts.length || parts.some((p) => Number.isNaN(p))) return null;
  return Math.round(parts.reduce((acc, p) => acc * 60 + p, 0) * 1000);
}

/** Matches [1:05], [0:41-0:58], [1:02:03 - 1:05:00] in answer text. */
export const TIME_TAG = /\[(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?))?\]/g;

export const STAGE_LABELS: Record<string, string> = {
  normalize: "Preparing video & audio",
  chunk: "Splitting into chunks",
  transcribe: "Transcribing speech",
  shots: "Detecting shots",
  caption: "Describing frames",
  merge: "Building the timeline",
  scenes: "Writing scene descriptions",
  embed: "Building the search index",
  summarize: "Summarizing chapters",
};
