// Thin horizontal progress bar — shared by LiveProgress and the
// KnowledgeGraph empty-state placeholder.
export function ProgressBar({
  value,
  pulsing = false,
  className = '',
}: {
  value: number;
  pulsing?: boolean;
  className?: string;
}) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={`h-1.5 rounded-full bg-gray-100 overflow-hidden ${className}`}>
      <div
        className={
          'h-full bg-brand-600 transition-[width] duration-500'
          + (pulsing ? ' animate-pulse-slow' : '')
        }
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}
