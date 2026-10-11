"use client";

import { useEffect, useRef, useState } from "react";
import { askStream, type AskResult, type ToolCall } from "@/lib/api";
import TimedText from "./TimedText";
import TimeChip from "./TimeChip";

interface Message {
  role: "user" | "bot";
  text: string;
  status?: string;
  steps?: ToolCall[];
  result?: AskResult;
  error?: string;
  pending?: boolean;
}

const SUGGESTIONS = ["What is this video about?", "What genre is it?", "Who are the main characters?",
  "What happens in the first minute?", "How does it end?"];

const STEP_LABELS: Record<string, string> = {
  search_moments: "searching moments", find_scenes_by_tag: "finding tagged scenes", get_range: "reading a time range",
  search_dialogue: "searching dialogue", get_movie_summary: "reading the summary",
  get_chapter_summaries: "reading chapters", get_characters: "looking up characters",
};

export default function Chat({ videoId, onSeek }: { videoId: string; onSeek: (ms: number) => void }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const busy = messages.some((m) => m.pending);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages]);

  const updateLast = (fn: (m: Message) => Message) =>
    setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));

  async function send(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    setInput("");
    setMessages((ms) => [...ms, { role: "user", text: q }, { role: "bot", text: "", steps: [], pending: true, status: "sending" }]);
    try {
      await askStream(videoId, q, {
        onStatus: (status) => updateLast((m) => ({ ...m, status })),
        onStep: (step) => updateLast((m) => ({ ...m, steps: [...(m.steps ?? []), step] })),
        onToken: (t) => updateLast((m) => ({ ...m, text: m.text + t, status: undefined })),
        onReset: () => updateLast((m) => ({ ...m, text: "" })),
        onAnswer: (result) => updateLast((m) => ({ ...m, text: result.answer, result, pending: false, status: undefined })),
        onError: (error) => updateLast((m) => ({ ...m, error, pending: false })),
      });
    } catch (e) {
      updateLast((m) => ({ ...m, error: (e as Error).message, pending: false }));
    }
    updateLast((m) => (m.pending ? { ...m, pending: false, error: m.error ?? "The connection closed early." } : m));
  }

  return (
    <div className="panel chat">
      <h2>Ask about this video</h2>
      <div className="messages">
        {messages.length === 0 && (
          <div className="muted small">
            Ask anything: a time range (&quot;what happens between 1:00 and 2:00?&quot;), scenes (&quot;show the fight scenes&quot;),
            quotes, characters, genre, the ending… Answers take ~20–90 s on a local model.
            <div className="suggest">{SUGGESTIONS.map((s) => <button key={s} onClick={() => send(s)}>{s}</button>)}</div>
          </div>
        )}
        {messages.map((m, i) => m.role === "user" ? (
          <div key={i} className="msg user">{m.text}</div>
        ) : (
          <div key={i} className="msg bot">
            {!!m.steps?.length && (
              <div className="steps">{m.steps.map((s) => STEP_LABELS[s.action] ?? s.action).join(" → ")}</div>
            )}
            {m.text ? <TimedText text={m.text} onSeek={onSeek} />
              : m.pending && <span className="muted dots">{m.status === "sending" ? "Sending" : m.status ?? "Thinking"}</span>}
            {m.error && <div className="error small">{m.error}</div>}
            {!!m.result?.sources.length && (
              <div className="sources">
                <span className="muted small">Sources</span>
                {m.result.sources.map((s, j) => (
                  <div key={j}><TimeChip start={s.start_ms} end={s.end_ms} onSeek={onSeek} /> <span className="muted">{s.label}</span></div>
                ))}
              </div>
            )}
            {m.result && (
              <div className="muted small" style={{ marginTop: 6 }}>
                {m.result.model} · {m.result.seconds.toFixed(0)} s
                {m.result.unverified_times.length ? ` · unverified times: ${m.result.unverified_times.join(", ")}` : ""}
              </div>
            )}
          </div>
        ))}
        <div ref={end} />
      </div>
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(input); }}>
        <input className="input" placeholder={busy ? "Answering…" : "Ask a question"} value={input}
          onChange={(e) => setInput(e.target.value)} disabled={busy} maxLength={1000} />
        <button className="btn primary" disabled={busy || !input.trim()}>Ask</button>
      </form>
    </div>
  );
}
