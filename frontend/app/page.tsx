"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { API_URL, listVideos, type VideoSummary } from "@/lib/api";
import { fmt } from "@/lib/time";
import ProgressBar from "@/components/ProgressBar";
import Uploader from "@/components/Uploader";

export default function Library() {
  const [videos, setVideos] = useState<VideoSummary[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const load = () => listVideos().then((v) => alive && (setVideos(v), setError("")))
      .catch(() => alive && setError(`Can't reach the backend at ${API_URL}. Start it with: uvicorn app.main:app --port 8000`));
    load();
    const t = setInterval(load, 5000);          // refresh statuses while videos process
    return () => { alive = false; clearInterval(t); };
  }, []);

  return (
    <>
      <h1>Your videos</h1>
      <Uploader />
      {error && <p className="error">{error}</p>}
      {videos && videos.length === 0 && <p className="muted">No videos yet: upload one above.</p>}
      <div className="grid">
        {videos?.map((v) => (
          <Link key={v.video_id} href={`/videos/${v.video_id}`} className="card">
            <div className="title" title={v.title}>{v.title}</div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <span className={`badge ${v.status}`}>{v.status === "done" ? "ready" : v.status}</span>
              <span className="muted small">{v.duration_ms ? fmt(v.duration_ms) : ""}</span>
            </div>
            {v.status !== "done" && <ProgressBar value={v.overall} status={v.status} />}
          </Link>
        ))}
      </div>
    </>
  );
}
