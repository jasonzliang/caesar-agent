/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow a second instance on the same machine to build/serve into a
  // sibling directory (e.g. ".next-b") so two `next start` processes
  // don't race on the same build output. launch.sh exports
  // NEXT_DIST_DIR=.next-${CAESAR_INSTANCE_ID} when an instance ID is set;
  // legacy single-instance deployments fall back to the standard ".next".
  distDir: process.env.NEXT_DIST_DIR || '.next',
  // Disable Next's built-in gzip middleware. It otherwise applies to every
  // response, including text/event-stream — and buffers small SSE
  // heartbeats (~10 bytes per ping) inside the gzip window indefinitely,
  // so heartbeats never reach the client. That makes long-lived SSE
  // connections appear idle to downstream proxies (Tailscale Funnel,
  // browser auto-reconnect) which then close them periodically. Server-
  // side compression is the wrong layer for this stack anyway — if you
  // need response compression, put it in front of Next, not inside it.
  compress: false,
  experimental: {
    // Next's rewrite proxy uses an underlying http-proxy that idles upstream
    // sockets at 30s by default. With a 15s SSE heartbeat we'd usually be
    // fine, but combined with gzip buffering (see compress:false above) the
    // upstream socket can look idle from Next's perspective. 0 disables the
    // proxy-side idle timeout entirely; FastAPI's own keepalive + our SSE
    // ping=15 are sufficient to keep the connection healthy.
    proxyTimeout: 0,
  },
  // Single-origin design: the browser only ever talks to :3000. /api/* is
  // proxied to the FastAPI process server-side, so there's no CORS / mixed
  // origin issue for clients on a remote laptop.
  //
  // Read API_INTERNAL_URL inside `rewrites()` rather than at module top so
  // the value is resolved at server-start time (where launch.sh sets it),
  // not at build time (where the env var isn't set and the fallback below
  // gets baked in).
  async rewrites() {
    const apiInternalUrl =
      process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
    return [
      { source: '/api/:path*', destination: `${apiInternalUrl}/:path*` },
    ];
  },
};

export default nextConfig;
