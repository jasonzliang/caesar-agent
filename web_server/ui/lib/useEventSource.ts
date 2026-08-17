'use client';

import { useEffect, useRef, useState } from 'react';

export type LiveEvent = {
  id?: number;
  event: string;
  payload: Record<string, unknown>;
  timestamp?: string;
};

type Status = 'connecting' | 'open' | 'closed' | 'error';

/**
 * Subscribe to a Server-Sent Events stream and accumulate every event.
 *
 * - Auto-reconnect is handled by the browser's native EventSource.
 * - Events are de-duplicated by `id` so a reconnect doesn't double-render.
 * - The hook returns `events` in arrival order; consumers can derive any
 *   higher-level state from it.
 */
export function useEventSource(url: string | null) {
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [status, setStatus] = useState<Status>('connecting');
  const seenIds = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!url) {
      // Terminal runs pass null; mark closed so the header indicator doesn't stick on "Connecting…".
      setStatus('closed');
      return;
    }
    setEvents([]);
    seenIds.current = new Set();
    setStatus('connecting');

    const es = new EventSource(url);

    const onMessage = (raw: MessageEvent<string>) => {
      // Guard against frames with empty / undefined data — sse-starlette's
      // heartbeats and any future ping events shouldn't crash the parser.
      const text = raw?.data;
      if (text == null || text === '' || text === 'undefined') return;
      try {
        const data = JSON.parse(text) as LiveEvent;
        // Drop heartbeat pings — they have no payload of interest and would
        // otherwise grow `events` unboundedly across long sessions.
        if (data.event === 'ping') return;
        if (typeof data.id === 'number') {
          if (seenIds.current.has(data.id)) return;
          seenIds.current.add(data.id);
        }
        setEvents((prev) => [...prev, data]);
      } catch (e) {
        // Don't promote parse failures to console.error — they're noisy in
        // dev tools and almost always indicate a benign keepalive frame.
        console.warn('Skipped non-JSON SSE frame:', text.slice(0, 80));
      }
    };

    // Debounce 'error' status: EventSource auto-retries every ~3s; if
    // onopen fires within the window the badge never shows.
    let errorTimer: ReturnType<typeof setTimeout> | null = null;
    const clearErrorTimer = () => {
      if (errorTimer) {
        clearTimeout(errorTimer);
        errorTimer = null;
      }
    };
    es.onopen = () => {
      clearErrorTimer();
      setStatus('open');
    };
    es.onmessage = onMessage;
    // FastAPI / sse-starlette emits typed events; the browser only fires
    // listeners that match the `event:` name, so each known type needs an
    // explicit subscription. Keep this list in sync with EVENT_TYPES in
    // api/app/job_runner.py, plus 'ping' (heartbeat the server emits via
    // sse-starlette directly, not through EVENT_TYPES).
    for (const t of [
      'log',
      'iteration',
      'synthesis_progress',
      'graph_update',
      'draft_complete',
      'cost_update',
      'resumed',
      'done',
      'error',
      'ping',
    ]) {
      es.addEventListener(t, onMessage as EventListener);
    }
    es.onerror = () => {
      // `onerror` fires in two cases: (a) the server cleanly closed the
      // stream — readyState=CLOSED, the browser won't retry; or (b) a
      // network drop / server restart — readyState=CONNECTING, the browser
      // is auto-retrying. Clean closes flip to 'closed' (silent) instantly.
      if (es.readyState === EventSource.CLOSED) {
        clearErrorTimer();
        setStatus('closed');
        return;
      }
      if (errorTimer != null) return;
      // This debounce = how long a reconnect gap must last before we tell the
      // user the live stream is down. Trade-off in both directions:
      //  - too short: Tailscale Funnel drops streaming connections every
      //    10-40s regardless of byte flow, reconnecting in ~3-10s (handshakes
      //    spike higher), so a tight window flashes a false badge on every
      //    normal, self-healing reconnect (missed events replay via
      //    Last-Event-ID, so nothing is actually lost);
      //  - too long: a GENUINELY dead stream (server down, run crashed, no
      //    reconnect at all) keeps showing green "Live" for the whole window,
      //    so the badge lies about liveness.
      // 30s is ~3x the funnel's ~10s handshake ceiling — clears essentially
      // all false badges while still surfacing a real outage within ~30s.
      // Going much beyond this makes the badge untrustworthy, so treat it as
      // the ceiling, not a knob to keep raising.
      errorTimer = setTimeout(() => {
        errorTimer = null;
        if (es.readyState !== EventSource.CLOSED) setStatus('error');
      }, 30000);
    };

    return () => {
      clearErrorTimer();
      es.close();
      setStatus('closed');
    };
  }, [url]);

  return { events, status };
}
