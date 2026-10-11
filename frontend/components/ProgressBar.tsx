export default function ProgressBar({ value, status }: { value: number; status?: string }) {
  const pct = Math.round(Math.max(0, Math.min(value, 1)) * 100);
  return (
    <div className={`bar ${status ?? ""}`} role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${status === "failed" ? 100 : pct}%` }} />
    </div>
  );
}
