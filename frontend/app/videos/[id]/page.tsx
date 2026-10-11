"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { getVideo, streamUrl, type VideoDetail } from "@/lib/api";
import { fmt } from "@/lib/time";
import Chat from "@/components/Chat";
import InfoTabs from "@/components/InfoTabs";
import ProcessingView from "@/components/ProcessingView";

export default function WatchPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [video, setVideo] = useState<VideoDetail | null>(null);
  const [error, setError] = useState("");
  const [currentMs, setCurrentMs] = useState(0);
  const player = useRef<HTMLVideoElement>(null);

  const load = useCallback(() => {
    getVideo(id).then(setVideo).catch((e) => setError((e as Error).message));
  }, [id]);
  useEffect(load, [load]);

  /** Jump the player to a moment (used by every time chip). */
  const seek = useCallback((ms: number) => {
    const v = player.current;
    if (!v) return;
    v.currentTime = ms / 1000;
    v.play().catch(() => { /* autoplay blocked: the position is still set */ });
    v.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, []);

  if (error) return <p className="error" style={{ marginTop: 24 }}>{error} · <Link href="/">back</Link></p>;
  if (!video) return <p className="muted" style={{ marginTop: 24 }}>Loading…</p>;

  return (
    <>
      <h1 style={{ display: "flex", gap: 12, alignItems: "baseline" }}>
        <Link href="/" className="muted">←</Link>
        <span>{video.title}</span>
        {video.media.duration_ms && <span className="muted small">{fmt(video.media.duration_ms)}</span>}
      </h1>

      {!video.ready ? (
        <ProcessingView videoId={id} initial={video.progress} onReady={load} />
      ) : (
        <div className="watch">
          <div>
            <video ref={player} src={streamUrl(id)} controls preload="metadata"
              onTimeUpdate={(e) => setCurrentMs(Math.floor(e.currentTarget.currentTime * 1000))} />
            <InfoTabs videoId={id} currentMs={currentMs} onSeek={seek} />
          </div>
          <Chat videoId={id} onSeek={seek} />
        </div>
      )}
    </>
  );
}
