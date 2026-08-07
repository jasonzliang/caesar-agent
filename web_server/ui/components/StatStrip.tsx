// Compact horizontal credibility stats — sits right under the tagline.
// Numbers match the docs site at https://jasonzliang.github.io/caesar-agent/.
export function StatStrip(): React.JSX.Element {
  return (
    <div className="rounded-xl border border-brand-200 bg-brand-50 px-5 py-3 flex flex-wrap items-baseline justify-center gap-x-5 gap-y-1.5 text-xs text-gray-600">
      <Stat value="26.96 / 30" label="creativity score" />
      <Sep />
      <Stat value={<>+13&ndash;23%</>} label="over top deep-research baselines" />
      <Sep />
      <Stat value="δ ≥ 0.76" label="Cliff's Δ (large effect)" />
    </div>
  );
}

function Stat({ value, label }: { value: React.ReactNode; label: string }): React.JSX.Element {
  return (
    <span className="whitespace-nowrap">
      <span className="font-semibold text-brand-800 tabular-nums">{value}</span>{' '}
      <span className="text-gray-500">{label}</span>
    </span>
  );
}

function Sep(): React.JSX.Element {
  return <span className="text-gray-300 select-none">·</span>;
}
