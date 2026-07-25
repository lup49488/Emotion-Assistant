import 'katex/dist/katex.min.css'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

function normalizeMarkdownSegment(segment) {
  return segment
    .replace(/<br\s*\/?>/gi, '  \n')
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, formula) => `\n\n$$\n${formula.trim()}\n$$\n\n`)
    .replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, (_, formula) => `$${formula.trim()}$`)
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
