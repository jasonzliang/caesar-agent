import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';

// Match the renderer's Markdown grammar, but not remark-math: input dollars
// are currency. Only explicit LaTeX delimiters below introduce math.
const markdown = unified().use(remarkParse).use(remarkGfm).freeze();
const verbatimTypes = new Set([
  'code', 'inlineCode', 'definition', 'image', 'imageReference', 'html',
]);

// Container prefix (indentation and blockquote markers) before a line's text.
const linePrefix = /^(?:[ \t]*>)*[ \t]*$/;

/**
 * `\[…\]` that opens a line becomes a `$$` fence carrying that line's
 * container prefix, so list items, blockquotes and paragraphs stay intact.
 * @param {string} body text between the delimiters
 * @param {string} prefix the opener line's indentation / `>` markers
 */
function displayMath(body, prefix) {
  const depth = (prefix.match(/>/g) ?? []).length;
  // Each line already carries (up to) this prefix in the source, or is lazy.
  const strip = new RegExp(`^(?:[ \\t]*>){0,${depth}}[ \\t]*`);
  const lines = body.split(/\r?\n/).map((line, i) => (i ? line.replace(strip, '') : line).trim());
  while (lines.length && !lines[0]) lines.shift();
  while (lines.length && !lines.at(-1)) lines.pop();
  return `$$\n${[...lines, '$$'].map((line) => prefix + line).join('\n')}`;
}

/**
 * Normalize citations, LaTeX delimiters and currency only in Markdown prose.
 * Parser source positions protect all CommonMark/GFM code forms and links.
 * @param {string} src
 * @param {Iterable<number | string>} [sourceIds] ids in the sources list;
 *   when given, only `[n]` markers naming these ids become citation links.
 * @returns {string}
 */
export function prepareMarkdown(src, sourceIds) {
  const validIds = sourceIds == null ? null : new Set(Array.from(sourceIds, String));
  const protectedRanges = [];
  const linkLabels = [];
  const tableCells = [];
  const paragraphs = [];

  function protect(node) {
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (node.type === 'tableCell') tableCells.push([start, end]);
    if (node.type === 'paragraph') paragraphs.push([start, end, node.position.start.column]);
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
  const tokens = /\\\(((?:\\[\s\S]|[^\\])*?)\\\)|\\\[((?:\\[\s\S]|[^\\])*?)\\\]|\\[\s\S]|\$|\[[ \t]*(\d+(?:[ \t]*,[ \t]*\d+)*)[ \t]*\]/g;
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
    let excess = 0;
    if (inlineMath !== undefined) replacement = `$${inlineMath}$`;
    else if (blockMath !== undefined) {
      let prefix = src.slice(src.lastIndexOf('\n', position - 1) + 1, position);
      if (!linePrefix.test(prefix) || tableCells.some(([a, b]) => position >= a && position < b)) {
        // Mid-line (prose, table cell, after a list marker): a block here
        // would split the line, so render it inline.
        replacement = `$${blockMath}$`;
      } else {
        // Paragraph continuation may be indented arbitrarily deep, but a fence
        // 4+ columns past its container is indented code: align it with the
        // paragraph's own column instead.
        const column = paragraphs.find(([a, b]) => position >= a && position < b)?.[2];
        if (column != null && prefix.length > column - 1) {
          excess = prefix.length - (column - 1);
          prefix = prefix.slice(0, column - 1);
        }
        replacement = displayMath(blockMath, prefix);
        // A closing fence must end its line; move trailing text to the next.
        let end = tokens.lastIndex;
        while (src[end] === ' ' || src[end] === '\t') end++;
        if (end < src.length && src[end] !== '\n' && src[end] !== '\r') {
          replacement += `\n${prefix}`;
          tokens.lastIndex = end;
        }
      }
    } else if (token === '$') replacement = '\\$';
    else if (ids !== undefined && !linkLabels.some(([a, b]) => position >= a && position < b)) {
      const numbers = ids.split(',').map((id) => id.trim());
      // Unknown ids ([2024], [0], interval [0, 1]) are not citations.
      if (!validIds || numbers.every((number) => validIds.has(number))) {
        replacement = numbers.map((number) => `[${number}](#source-${number})`).join('');
      }
    }
    out += src.slice(cursor, position - excess) + replacement;
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
