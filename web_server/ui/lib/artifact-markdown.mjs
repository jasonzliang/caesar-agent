import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';

// Match the renderer's Markdown grammar, but not remark-math: input dollars
// are currency. Only explicit LaTeX delimiters below introduce math.
const markdown = unified().use(remarkParse).use(remarkGfm).freeze();
const verbatimTypes = new Set([
  'code', 'inlineCode', 'definition', 'image', 'imageReference', 'html',
]);

/**
 * Normalize citations, LaTeX delimiters and currency only in Markdown prose.
 * Parser source positions protect all CommonMark/GFM code forms and links.
 * @param {string} src
 * @returns {string}
 */
export function prepareMarkdown(src) {
  const protectedRanges = [];
  const linkLabels = [];

  function protect(node) {
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (verbatimTypes.has(node.type)) {
      protectedRanges.push([start, end]);
      return;
    }
    if (node.type === 'link' || node.type === 'linkReference') {
      const children = node.children ?? [];
      const labelStart = children[0]?.position?.start.offset;
      const labelEnd = children.at(-1)?.position?.end.offset;
      // Autolinks are URLs throughout. Explicit link labels may contain math
      // or currency, but must never acquire nested citation links.
      if (src[start] !== '[' || labelStart == null || labelEnd == null) {
        protectedRanges.push([start, end]);
        return;
      }
      protectedRanges.push([start, labelStart], [labelEnd, end]);
      linkLabels.push([labelStart, labelEnd]);
    }
    for (const child of node.children ?? []) protect(child);
  }
  protect(markdown.parse(src));
  protectedRanges.sort((a, b) => a[0] - b[0]);

  // The parser decides where code and link syntax begin/end. Token matching
  // only handles our prose conventions; citations use horizontal whitespace
  // so a malformed prefix cannot span across blank lines into a code block.
  const tokens = /\\\(((?:\\[\s\S]|[^\\])*?)\\\)|\\\[((?:\\[\s\S]|[^\\])*?)\\\]|\\[\s\S]|\$|\[(\d+(?:[ \t]*,[ \t]*\d+)*)\]/g;
  let out = '';
  let cursor = 0;
  let rangeIndex = 0;
  let match;
  while ((match = tokens.exec(src))) {
    const [token, inlineMath, blockMath, ids] = match;
    const position = match.index;
    while (rangeIndex < protectedRanges.length && protectedRanges[rangeIndex][1] <= position) rangeIndex++;
    const range = protectedRanges[rangeIndex];
    if (range && range[0] <= position) {
      // Skip the whole protected node, not just the token: a literal math
      // opener inside code/a URL must not consume subsequent prose.
      tokens.lastIndex = range[1];
      continue;
    }
    let replacement = token;
    if (inlineMath !== undefined) replacement = `$${inlineMath}$`;
    else if (blockMath !== undefined) replacement = `\n\n$$\n${blockMath.trim()}\n$$\n\n`;
    else if (token === '$') replacement = '\\$';
    else if (ids !== undefined && !linkLabels.some(([a, b]) => position >= a && position < b)) {
      replacement = ids.split(',').map((id) => {
        const number = id.trim();
        return `[${number}](#source-${number})`;
      }).join('');
    }
    out += src.slice(cursor, position) + replacement;
    cursor = tokens.lastIndex;
  }
  return out + src.slice(cursor);
}

/**
 * Plain text of the actual rendered <pre><code> children, for the clipboard.
 * @param {import('react').ReactNode} node
 * @returns {string}
 */
export function extractText(node) {
  if (node == null || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(extractText).join('');
  if (typeof node === 'object' && 'props' in node) {
    return extractText(node.props?.children);
  }
  return '';
}
