"use client";

import { Fragment } from "react";
import { TIME_TAG, parseTime } from "@/lib/time";
import TimeChip from "./TimeChip";

/** Renders text, turning [1:05] / [0:41-0:58] into clickable time chips. */
export default function TimedText({ text, onSeek }: { text: string; onSeek: (ms: number) => void }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(TIME_TAG)) {
    const start = parseTime(m[1]);
    if (start === null || m.index === undefined) continue;
    parts.push(text.slice(last, m.index));
    const end = m[2] ? parseTime(m[2]) ?? undefined : undefined;
    parts.push(<TimeChip key={m.index} start={start} end={end} onSeek={onSeek} />);
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return <>{parts.map((p, i) => <Fragment key={i}>{p}</Fragment>)}</>;
}
