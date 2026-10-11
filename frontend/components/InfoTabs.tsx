"use client";

import { useEffect, useState } from "react";
import { getChapters, getProfile, getScenes, type Chapter, type Profile, type Scene } from "@/lib/api";
import TimeChip from "./TimeChip";

type Tab = "scenes" | "chapters" | "profile";

/** Scenes / chapters / profile under the player. The scene playing now is highlighted. */
export default function InfoTabs({ videoId, currentMs, onSeek }:
  { videoId: string; currentMs: number; onSeek: (ms: number) => void }) {
  const [tab, setTab] = useState<Tab>("scenes");
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([getScenes(videoId), getChapters(videoId), getProfile(videoId)])
      .then(([s, c, p]) => { setScenes(s); setChapters(c); setProfile(p); })
      .catch((e) => setError((e as Error).message));
  }, [videoId]);

  return (
    <>
      <div className="tabs">
        {(["scenes", "chapters", "profile"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
            {t[0].toUpperCase() + t.slice(1)}{t === "scenes" && scenes.length ? ` (${scenes.length})` : ""}
          </button>
        ))}
      </div>
      {error && <p className="error small">{error}</p>}

      {tab === "scenes" && (
        <div className="list">
          {scenes.map((s) => {
            const active = currentMs >= s.start_ms && currentMs < s.end_ms;
            const tags = Object.entries(s.tags).filter(([, v]) => v >= 0.5).sort((a, b) => b[1] - a[1]);
            return (
              <div key={s.scene} className={`item ${active ? "active" : ""}`}>
                <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <TimeChip start={s.start_ms} end={s.end_ms} onSeek={onSeek} />
                  <b>{s.title}</b>
                  {s.mood && <span className="muted small">{s.mood}</span>}
                </div>
                {s.description && <div className="small" style={{ marginTop: 4 }}>{s.description}</div>}
                {tags.map(([t, v]) => (
                  <span key={t} className="tag" title={s.tag_evidence?.[t] ?? ""}>{t.replace(/_/g, " ")} {v.toFixed(1)}</span>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {tab === "chapters" && (
        <div className="list">
          {chapters.map((c) => (
            <div key={c.chapter} className="item">
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <TimeChip start={c.start_ms} end={c.end_ms} onSeek={onSeek} />
                <b>{c.title || `Chapter ${c.chapter}`}</b>
              </div>
              <div className="small" style={{ marginTop: 4 }}>{c.error ? <span className="error">{c.error}</span> : c.summary}</div>
            </div>
          ))}
        </div>
      )}

      {tab === "profile" && profile && (
        <div className="panel small">
          {profile.error ? <p className="error">{profile.error}</p> : (
            <>
              <p><b>Type:</b> {profile.video_type?.replace(/_/g, " ")}</p>
              <p><b>Genres:</b> {profile.genres?.map((g) => `${g.genre.replace(/_/g, " ")} (${g.confidence})`).join(", ") || "—"}</p>
              {profile.logline && <p><b>Logline:</b> {profile.logline}</p>}
              {profile.plot && <p><b>Plot:</b> {profile.plot}</p>}
              {profile.ending && <p><b>Ending:</b> {profile.ending}</p>}
              <p><b>Setting:</b> {profile.setting || "—"} · <b>Tone:</b> {profile.tone || "—"}</p>
              {!!profile.themes?.length && <p><b>Themes:</b> {profile.themes.join(", ")}</p>}
              {!!profile.characters?.length && (
                <>
                  <b>Characters</b> <span className="muted">(names come from speech recognition and may be misspelled)</span>
                  <ul>{profile.characters.map((c) => <li key={c.name}><b>{c.name}</b> ({c.role}): {c.description}</li>)}</ul>
                </>
              )}
            </>
          )}
        </div>
      )}
    </>
  );
}
