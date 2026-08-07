import { cn } from '@/lib/utils';
import type { RunStatus } from '@/lib/api';

const LABELS: Record<RunStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  // Transient: synthesis exited via cooperative shutdown; the server boot
  // resubmits it from scratch. Users typically only see this during the
  // restart gap or if a resubmit failed.
  interrupted: 'Restarting…',
};

// Reuse the running pill style for "interrupted" — it's a transient
// pre-resume state, not a terminal one, and a fresh CSS class would be
// noise. (If the resubmit fails, the row flips to "failed".)
const CSS_STATUS: Record<RunStatus, string> = {
  queued: 'queued',
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  interrupted: 'running',
};

export function StatusBadge({ status }: { status: RunStatus }) {
  return (
    <span className={cn('pill', `pill-${CSS_STATUS[status] ?? status}`)}>
      <span className="dot" />
      {LABELS[status] ?? status}
    </span>
  );
}
