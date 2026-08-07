import type { RunMode } from '@/lib/api';

// Renders nothing for non-follow-up runs. A follow-up is a run with
// mode=explore|refine AND a parent_run_id — both must be set, since a
// fresh "new" run also has mode=new and the absence of parent_run_id.
export function FollowUpBadge({
  mode,
  parentRunId,
}: {
  mode?: RunMode | null;
  parentRunId?: string | null;
}) {
  const isFollowUp = (mode === 'explore' || mode === 'refine') && !!parentRunId;
  if (!isFollowUp) return null;
  return (
    <span className="pill pill-followup" title="Follow-up run">
      Follow-up
    </span>
  );
}
