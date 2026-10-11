"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { importLocal, uploadVideo } from "@/lib/api";
import ProgressBar from "./ProgressBar";

const ACCEPT = ".mp4,.mkv,.mov,.avi,.webm,.m4v";

/** Drag & drop / pick a file → upload with progress → open the video page. */
export default function Uploader() {
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const abortRef = useRef<() => void>(() => {});
  const [over, setOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [localPath, setLocalPath] = useState("");

  async function start(f: File) {
    setFile(f); setError(""); setProgress(0);
    const { promise, abort } = uploadVideo(f, setProgress);
    abortRef.current = abort;
    try {
      const { video_id } = await promise;
      router.push(`/videos/${video_id}`);
    } catch (e) {
      setError((e as Error).message); setFile(null);
    }
  }

  async function registerLocal() {
    setError("");
    try {
      const { video_id } = await importLocal(localPath.trim());
      router.push(`/videos/${video_id}`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="panel">
      <div
        className={`drop ${over ? "over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !file) start(f); }}
      >
        {file ? (
          <>
            <div style={{ marginBottom: 10 }}>Uploading <b>{file.name}</b> — {Math.round(progress * 100)}%</div>
            <ProgressBar value={progress} />
            <button className="btn" style={{ marginTop: 12 }} onClick={() => abortRef.current()}>Cancel</button>
          </>
        ) : (
          <>
            <div style={{ fontSize: 16, marginBottom: 6 }}>Drop a movie or video here</div>
            <div className="muted small" style={{ marginBottom: 12 }}>MP4, MKV, MOV, AVI, WEBM · processing starts automatically</div>
            <button className="btn primary" onClick={() => input.current?.click()}>Choose file</button>
            <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => e.target.files?.[0] && start(e.target.files[0])} />
          </>
        )}
      </div>
      <details style={{ marginTop: 12 }}>
        <summary className="muted small" style={{ cursor: "pointer" }}>File already on this PC? Add it without uploading</summary>
        <div className="composer">
          <input className="input" placeholder="D:\Movies\film.mkv" value={localPath} onChange={(e) => setLocalPath(e.target.value)} />
          <button className="btn" disabled={!localPath.trim()} onClick={registerLocal}>Add</button>
        </div>
      </details>
      {error && <p className="error small">{error}</p>}
    </div>
  );
}
