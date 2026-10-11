"use client";

import { useEffect, useState } from "react";
import { restartProcessing, watchProgress, type Progress } from "@/lib/api";
import { STAGE_LABELS } from "@/lib/time";
import ProgressBar from "./ProgressBar";

/** Live pipeline progress (Server-Sent Events). Calls onReady when processing finishes. */
export default function ProcessingView({ videoId, initial, onReady }:
  { videoId: string; initial: Progress; onReady: () => void }) {
  const [p, setP] = useState<Progress>(initial);
  const [lost, setLost] = useState(false);

  useEffect(() => {
    const stop = watchProgress(videoId, (next) => {
      setP(next); setLost(false);
      if (next.status === "done") onReady();
    }, () => setLost(true));
    return stop;
  }, [videoId, onReady]);

  async function retry() {
    setP(await restartProcessing(videoId));
    location.reload();
  }

  return (
    <div className="panel" style={{ maxWidth: 640, margin: "24px 0" }}>
      <h2>{p.status === "failed" ? "Processing failed" : p.status === "queued" ? "Waiting in the queue" : "Processing"}</h2>
      <p className="muted small" style={{ marginTop: 0 }}>
        {p.queue_position ? `Position ${p.queue_position} in the queue. ` : ""}
        {p.message}{lost ? " (reconnecting…)" : ""}
      </p>
      <ProgressBar value={p.overall} status={p.status} />
      <p className="small muted">{Math.round(p.overall * 100)}% overall · long movies take a while (captions and scene descriptions run on the GPU)</p>
      <ul className="stages">
        {Object.entries(p.stages).map(([name, status]) => {
          const live = p.stage === name && status !== "done";
          return (
            <li key={name}>
              <span className={`dot ${live ? "running" : status}`} />
              <span>{STAGE_LABELS[name] ?? name}</span>
              {live && <span className="muted small">{Math.round(p.fraction * 100)}%</span>}
            </li>
          );
        })}
      </ul>
      {p.status === "failed" && (
        <>
          <p className="error small">{p.error}</p>
          <button className="btn" onClick={retry}>Retry (finished steps are kept)</button>
        </>
      )}
    </div>
  );
}
