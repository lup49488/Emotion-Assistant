import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

function escapeTablePipes(formula) {
  // A raw pipe inside math would otherwise be parsed as the next GFM table cell.
  // Matching `\\.` first keeps already-escaped sequences intact and lets every
  // bare pipe be replaced, including consecutive ones.
  return formula.replace(/\\.|\|/g, (match) => (match === '|' ? '\\vert ' : match))
}

function normalizeTableRowMath(line) {
  return line
    // Block math is not valid inside a table cell; retain the formula inline.
    .replace(/\$\$([^\n]+?)\$\$/g, (_, formula) => `$${escapeTablePipes(formula.trim())}$`)
    .replace(/(^|[^$])\$([^$\n]+?)\$(?!\$)/g, (_, prefix, formula) => (
      `${prefix}$${escapeTablePipes(formula)}$`
    ))
}

function normalizeMarkdownSegment(segment) {
  return segment
    .replace(/<br\s*\/?>/gi, '  \n')
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, formula) => `\n\n$$\n${formula.trim()}\n$$\n\n`)
    .replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, (_, formula) => `$${formula.trim()}$`)
    .split('\n')
    // Only table rows need pipe escaping. Elsewhere a raw pipe is harmless, and
    // rewriting it would break valid math such as \begin{array}{c|c}.
    .map((line) => (line.trimStart().startsWith('|') ? normalizeTableRowMath(line) : line))
    .join('\n')
}

function normalizeAssistantMarkdown(content) {
  return String(content || '')
    .split(/(```[\s\S]*?```)/g)
    .map((segment, index) => index % 2 === 0 ? normalizeMarkdownSegment(segment) : segment)
    .join('')
}

export default function AssistantMarkdown({ content }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={{ a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}>{normalizeAssistantMarkdown(content)}</ReactMarkdown>
}
