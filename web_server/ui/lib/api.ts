// Typed fetch wrappers. All paths are relative — the browser only ever talks
// to /api/*, which Next.js rewrites to the FastAPI server-side.

export type Preset = {
  id: string;
  label: string;
  description: string;
  estimated_cost_usd: number;
  estimated_time_min: number;
};

// An OpenAI model offered as a synthesis-model override (public mode).
export type ModelChoice = {
  id: string;
  input_per_mtok: number | null;
  output_per_mtok: number | null;
};

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';

export type RunMode = 'new' | 'explore' | 'refine';

export type RunSummary = {
  id: string;
  query: string;
  preset: string;
  status: RunStatus;
  parent_run_id: string | null;
  mode: RunMode;
  collection_name: string | null;
  owner_id: string | null;
  created_at: string;
  started_at: string | null;
  // Seconds burned in earlier attempts; runElapsedSec() adds it to the current span.
  elapsed_prior_s?: number;
  finished_at: string | null;
  total_cost_usd: number | null;
  graph_node_count: number | null;
  error_message: string | null;
  merged_query: string | null;
};

export type RunEvent = {
  id: number;
  timestamp: string;
  event: string;
  payload: Record<string, unknown>;
};

export type RunDetail = RunSummary & {
  events: RunEvent[];
  llm_model: string | null;
  graph_run_id: string | null;
  graph_progress_total: number | null;
  graph_progress_phase: 'quick_explore' | 'explore' | 'kb_ingest' | null;
};

export type GraphNode = {
  id: string;
  depth: number;
  insights: string | null;
  iteration: number | null;
  visit_count: number | null;
};

export type GraphEdge = {
  source: string;
  target: string;
  reason: string | null;
};

export type GraphOut = {
  iteration: number;
  starting_url: string | null;
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type SearchResultItem = { title: string; url: string; description: string };
export type SearchResultsOut = { results: SearchResultItem[] };

export type Citation = { index: number; url: string };

export type SynthesisOut = {
  draft: string;
  abstract: string;
  artifact: string;
  sources: Citation[];
  metadata: Record<string, unknown>;
  // Parent dir of the artifact file relative to run.repository — used to
  // resolve markdown image refs the synthesizer writes relative to the
  // artifact's location. Empty when the artifact is at repo root.
  artifact_dir: string;
};

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getRun: (id: string) =>
    fetch(`/api/runs/${id}`, { cache: 'no-store' }).then((r) => jsonOrThrow<RunDetail>(r)),

  // OpenAI models offered as synthesis-model overrides (public mode dropdown).
  getModels: () =>
    fetch('/api/models', { cache: 'no-store' }).then((r) => jsonOrThrow<ModelChoice[]>(r)),

  createRun: (
    query: string,
    preset: string,
    opts?: {
      parent_run_id?: string;
      mode?: RunMode;
      apiKey?: string;
      synthesisModel?: string;
      outputLength?: string;
    },
  ) =>
    fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        preset,
        ...(opts?.parent_run_id ? { parent_run_id: opts.parent_run_id } : {}),
        ...(opts?.mode ? { mode: opts.mode } : {}),
        // Public mode only: the user's OpenAI key rides the request body and is
        // never persisted server-side. Omitted entirely in shared/password mode.
        ...(opts?.apiKey ? { api_key: opts.apiKey } : {}),
        // Public mode only: synthesis-model override. Omitted → preset default.
        ...(opts?.synthesisModel ? { synthesis_model: opts.synthesisModel } : {}),
        // Public mode only: artifact word target. Omitted → preset default.
        ...(opts?.outputLength ? { output_length: Number(opts.outputLength) } : {}),
      }),
    }).then((r) => jsonOrThrow<RunSummary>(r)),

  // Restart a failed run in place (same id, same KB, resumes from its
  // checkpoint when one survived). Public mode must resend the user's key: the
  // server deletes the stored one the instant a run goes terminal.
  retryRun: (id: string, opts?: { apiKey?: string }) =>
    fetch(`/api/runs/${id}/retry`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts?.apiKey ? { api_key: opts.apiKey } : {}),
    }).then((r) => jsonOrThrow<RunSummary>(r)),

  getGraph: (id: string, iter: string | number = 'latest') =>
    fetch(`/api/runs/${id}/graph?iter=${iter}`, { cache: 'no-store' }).then((r) =>
      jsonOrThrow<GraphOut>(r),
    ),

  getSearchResults: (id: string) =>
    fetch(`/api/runs/${id}/search-results`, { cache: 'no-store' }).then((r) =>
      jsonOrThrow<SearchResultsOut>(r),
    ),

  getSynthesis: (id: string, draft: string | number = 'latest') =>
    fetch(`/api/runs/${id}/synthesis?draft=${draft}`, { cache: 'no-store' }).then((r) =>
      jsonOrThrow<SynthesisOut>(r),
    ),

  deleteRun: async (id: string): Promise<void> => {
    const res = await fetch(`/api/runs/${id}`, { method: 'DELETE' });
    if (res.ok) return;
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  },
};
