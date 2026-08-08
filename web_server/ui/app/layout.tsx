import type { Metadata } from 'next';
import './globals.css';
import { PublicModeProvider } from '@/lib/public-mode';
import { ApiKeyNavLink } from '@/components/ApiKeyNavLink';
import { AdminLink } from '@/components/AdminLink';

export const metadata: Metadata = {
  title: 'Caesar: Autonomous AI Research Agent',
  description:
    'Caesar is an autonomous AI research agent. Graph-based deep web exploration with adversarial answer synthesis. Outperforms GPT-5 and Gemini Deep Research on creative reasoning.',
  metadataBase: new URL('https://jasonzliang.github.io/caesar-agent/'),
  openGraph: {
    title: 'Caesar: Autonomous AI Research Agent',
    description:
      'Watch Caesar build a knowledge graph and refine its answer through adversarial self-critique.',
    type: 'website',
  },
};

type VersionInfo = {
  version: string | null;
  commit: string | null;
  uptimeSeconds: number | null;
  publicMode: boolean;
};

async function fetchVersion(): Promise<VersionInfo> {
  try {
    const apiUrl = process.env.API_INTERNAL_URL ?? 'http://127.0.0.1:8090';
    const res = await fetch(`${apiUrl}/version`, { cache: 'no-store' });
    if (!res.ok)
      return { version: null, commit: null, uptimeSeconds: null, publicMode: false };
    const data = (await res.json()) as {
      version?: string;
      commit?: string | null;
      uptime_seconds?: number | null;
      public_mode?: boolean;
    };
    const v = data.version;
    const version = v && !v.includes('unknown') ? v : null;
    return {
      version,
      commit: data.commit ?? null,
      uptimeSeconds: typeof data.uptime_seconds === 'number' ? data.uptime_seconds : null,
      publicMode: data.public_mode === true,
    };
  } catch {
    return { version: null, commit: null, uptimeSeconds: null, publicMode: false };
  }
}

function formatUptime(seconds: number): string {
  const hours = seconds / 3600;
  if (hours < 24) return `${Math.round(hours)}h`;
  return `${Math.round(hours / 24)}d`;
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const { version, commit, uptimeSeconds, publicMode } = await fetchVersion();
  return (
    <html lang="en">
      <body className="font-sans">
        <PublicModeProvider value={publicMode}>
        <div className="min-h-screen flex flex-col">
          <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-30">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
              <a href="/" className="flex items-center gap-2 group">
                <div className="w-8 h-8 rounded-lg bg-brand-800 text-white flex items-center justify-center font-bold">
                  C
                </div>
                <span className="font-semibold text-gray-900 tracking-tight">
                  Caesar
                </span>
                <span className="text-xs text-gray-500 hidden sm:inline">
                  Autonomous AI research agent
                </span>
              </a>
              <nav className="flex items-center gap-4 text-sm text-gray-600">
                {publicMode && <ApiKeyNavLink />}
                <a href="/runs" className="hover:text-gray-900">
                  Past Runs
                </a>
                <a
                  href="https://jasonzliang.github.io/caesar-agent/"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-gray-900"
                >
                  About
                </a>
                <a
                  href="https://www.cognizant.com/us/en/ai-lab/publications/ai-agents-creative-web-exploration"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-gray-900"
                >
                  Paper
                </a>
                <a
                  href="https://jasonzliang.github.io/"
                  target="_blank"
                  rel="noopener"
                  className="hover:text-gray-900"
                >
                  Author
                </a>
                {!publicMode && (
                  <a
                    href="mailto:jasonzliang@utexas.edu"
                    className="hover:text-gray-900"
                  >
                    Feedback
                  </a>
                )}
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="border-t border-gray-200 mt-12 py-6 text-sm text-gray-500">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-wrap items-center justify-between gap-2">
              <span>
                By{' '}
                <a
                  href="https://jasonzliang.github.io/"
                  target="_blank"
                  rel="noopener"
                  className="hover:underline hover:text-gray-900"
                >
                  Jason Liang
                </a>
                , Elliot Meyerson, Risto Miikkulainen (
                <a
                  href="https://www.cognizant.com/us/en/ai-lab"
                  target="_blank"
                  rel="noopener"
                  className="hover:underline hover:text-gray-900"
                >
                  Cognizant AI Lab
                </a>
                )
              </span>
              <span>
                <a
                  href="https://arxiv.org/abs/2604.20855"
                  target="_blank"
                  rel="noopener"
                  className="hover:underline"
                  title="Paper on arXiv"
                >
                  arXiv:2604.20855
                </a>
                {version && (
                  <>
                    {' · '}
                    <a
                      href="https://github.com/jasonzliang/caesar-agent"
                      target="_blank"
                      rel="noopener"
                      className="hover:underline"
                      title="Code on GitHub"
                    >
                      v{version}
                    </a>
                  </>
                )}
                {commit && (
                  <>
                    {' · '}
                    <a
                      href="https://github.com/jasonzliang/caesar-agent/commits/main/"
                      target="_blank"
                      rel="noopener"
                      className="hover:underline"
                      title="Commit history on GitHub"
                    >
                      {commit}
                    </a>
                  </>
                )}
                {uptimeSeconds != null && (
                  <>
                    {' · '}
                    <span title="Server uptime">
                      {formatUptime(uptimeSeconds)}
                    </span>
                  </>
                )}
                {publicMode && (
                  <>
                    {' · '}
                    <AdminLink />
                  </>
                )}
              </span>
            </div>
          </footer>
        </div>
        </PublicModeProvider>
      </body>
    </html>
  );
}
