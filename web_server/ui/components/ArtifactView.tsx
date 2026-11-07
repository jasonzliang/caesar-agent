'use client';

import { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { Check, Copy } from 'lucide-react';
import { extractText, prepareMarkdown } from '../lib/artifact-markdown.mjs';
import type { SynthesisOut } from '@/lib/api';
import 'katex/dist/katex.min.css';

// wrapEnabled is owned by the parent (RunPageClient) so the toggle button
// can live in the run toolbar next to Download PDF. ArtifactView just
// applies it as a class on <article>.
export function ArtifactView({
  synthesis,
  runId,
  wrapEnabled = false,
}: {
  synthesis: SynthesisOut;
  runId: string;
  wrapEnabled?: boolean;
}) {
  const transformedAbstract = useMemo(
    () => prepareMarkdown(synthesis.abstract ?? ''),
    [synthesis.abstract],
  );
  const transformedArtifact = useMemo(
    () => prepareMarkdown(synthesis.artifact),
    [synthesis.artifact],
  );
  // Embedded markdown images use relative paths like `images/<file>.png`;
  // the API serves them at /api/runs/{runId}/file/<artifact_dir>/<path>.
  // artifact_dir is empty for single-draft runs (artifact at repo root) and
  // the .synthesis.<ts> subdir for multi-draft runs.
  const mdComponents = useMemo(
    () => buildMdComponents(runId, synthesis.artifact_dir ?? ''),
    [runId, synthesis.artifact_dir],
  );

  return (
    <article className={`prose prose-lg prose-gray max-w-none answer-prose${wrapEnabled ? ' wrap-code' : ''}`}>
      {synthesis.abstract && (
        <aside className="not-prose mb-8 border border-gray-200 bg-brand-50/60 rounded-2xl px-5 py-4">
          <div className="text-[11px] uppercase tracking-[0.14em] font-semibold text-brand-800 mb-1.5">
            Abstract
          </div>
          <div className="prose prose-base prose-gray max-w-none text-gray-800 leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, { throwOnError: false }]]} components={mdComponents}>
              {transformedAbstract}
            </ReactMarkdown>
          </div>
        </aside>
      )}
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[[rehypeKatex, { throwOnError: false }]]} components={mdComponents}>
        {transformedArtifact}
      </ReactMarkdown>
    </article>
  );
}

function buildMdComponents(runId: string, artifactDir: string) {
  return {
    a({ href, children, ...rest }: React.AnchorHTMLAttributes<HTMLAnchorElement>) {
      if (typeof href === 'string' && href.startsWith('#source-')) {
        return (
          <a href={href} className="citation" {...rest}>
            {children}
          </a>
        );
      }
      return (
        <a href={href} {...rest}>
          {children}
        </a>
      );
    },
    // Fenced code blocks → <pre> with a copy button. Inline backticks emit
    // <code> with no <pre> wrapper and pass through unchanged.
    pre({ children, ...rest }: React.HTMLAttributes<HTMLPreElement>) {
      return <CodeBlockWithCopy {...rest}>{children}</CodeBlockWithCopy>;
    },
    // Markdown images. Relative paths point into the run's images/ subdir
    // (synthesizer writes `![Figure N](images/<file>.png)`); rewrite them
    // through the run-file route so the browser can fetch them. Absolute /
    // protocol URLs (http://, /api/..., data:) pass through unchanged.
    // Sized at 60% of the prose width and wrapped in an anchor so a click
    // opens the full-resolution image in a new tab.
    img({ src, alt }: React.ImgHTMLAttributes<HTMLImageElement>) {
      // ImgHTMLAttributes types src as string | Blob | undefined; markdown
      // only ever emits string srcs so we narrow defensively.
      const resolved = typeof src === 'string'
        ? resolveImgSrc(src, runId, artifactDir)
        : null;
      if (!resolved) return null;
      return (
        <a
          href={resolved}
          target="_blank"
          rel="noopener noreferrer"
          className="not-prose block my-6 text-center"
          aria-label={alt || 'Open image in new tab'}
        >
          <img
            src={resolved}
            alt={alt || ''}
            loading="eager"
            className="inline-block w-[70%] h-auto rounded-md border border-gray-200 shadow-sm hover:shadow-md transition-shadow cursor-zoom-in"
          />
        </a>
      );
    },
  };
}

function resolveImgSrc(src: string, runId: string, artifactDir: string): string | null {
  if (!src) return null;
  // Pass through absolute URLs, protocol-relative, data URLs, and anything
  // already routed through the API. Only relative paths get rewritten.
  if (/^(https?:|data:|\/\/|\/api\/)/i.test(src)) return src;
  if (src.startsWith('/')) return src;
  const clean = src.replace(/^\.\//, '');
  // Prepend artifactDir so e.g. `images/x.png` written by the synthesizer
  // inside a multi-draft `.synthesis.<ts>/` subdir resolves correctly.
  const path = artifactDir ? `${artifactDir}/${clean}` : clean;
  return `/api/runs/${encodeURIComponent(runId)}/file/${path}`;
}

function CodeBlockWithCopy({
  children,
  ...rest
}: React.HTMLAttributes<HTMLPreElement>) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      const text = extractText(children);
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // Older browsers / insecure-context clipboard denial — silently no-op;
      // the user can still select-and-copy manually.
    }
  };
  return (
    // not-prose strips the prose plugin's pre-padding rules so the absolute
    // button positioning is predictable; the inner <pre> re-establishes the
    // monospace styling we want.
    <div className="not-prose relative group my-4 print:my-4">
      <pre
        {...rest}
        className="overflow-x-auto rounded-xl border border-gray-200 bg-gray-50 text-gray-900 text-sm leading-relaxed p-4 print:bg-white print:border-gray-300"
      >
        {children}
      </pre>
      <button
        type="button"
        onClick={onCopy}
        aria-label={copied ? 'Copied' : 'Copy code'}
        title={copied ? 'Copied!' : 'Copy'}
        className="absolute top-2 right-2 inline-flex items-center gap-1 rounded-md border border-gray-200 bg-white/90 hover:bg-white px-2 py-1 text-xs text-gray-600 hover:text-gray-900 shadow-sm opacity-0 group-hover:opacity-100 focus-visible:opacity-100 transition-opacity print:hidden"
      >
        {copied ? (
          <>
            <Check size={12} className="text-green-600" />
            <span>Copied</span>
          </>
        ) : (
          <>
            <Copy size={12} />
            <span>Copy</span>
          </>
        )}
      </button>
    </div>
  );
}
