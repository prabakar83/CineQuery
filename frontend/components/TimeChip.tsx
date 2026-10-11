"use client";

import { fmt } from "@/lib/time";

/** A clickable timestamp: jumps the player to start_ms. */
export default function TimeChip({ start, end, onSeek }: { start: number; end?: number; onSeek: (ms: number) => void }) {
  return (
    <button className="chip" onClick={() => onSeek(start)} title="Jump to this moment">
      ▶ {fmt(start)}{end !== undefined && end > start ? `–${fmt(end)}` : ""}
    </button>
  );
}
