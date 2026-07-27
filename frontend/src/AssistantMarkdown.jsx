import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

function escapeTablePipes(formula) {
  // A raw pipe inside math would otherwise be parsed as the next GFM table cell.
  return formula.replace(/(^|[^\\])\|/g, (_, prefix) => `${prefix}\\vert `)
}

function normalizeTableMath(segment) {
  return segment
    .split('\n')
    .map((line) => {
      if (!line.trimStart().startsWith('|')) return line
      // Block math is not valid inside a table cell; retain the formula inline.
      return line.replace(/\$\$([^\n]+?)\$\$/g, (_, formula) => `$${formula.trim()}$`)
    })
    .join('\n')
}

function normalizeMarkdownSegment(segment) {
  const normalized = segment
    .replace(/<br\s*\/?>/gi, '  \n')
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, formula) => `\n\n$$\n${formula.trim()}\n$$\n\n`)
    .replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, (_, formula) => `$${formula.trim()}$`)

  return normalizeTableMath(normalized)
    .replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => `$$${escapeTablePipes(formula)}$$`)
    .replace(/(^|[^$])\$([^$\n]+?)\$(?!\$)/gm, (_, prefix, formula) => (
      `${prefix}$${escapeTablePipes(formula)}$`
    ))
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
