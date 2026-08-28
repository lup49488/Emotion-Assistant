#!/usr/bin/env node
// Design-system adherence checks from the Nocturne _ds bundle's _adherence.oxlintrc.json.
//
// They ship as oxlint `no-restricted-syntax` selectors, which oxlint does not
// implement, so they run here instead. The scope matches the original selectors:
// string and template literals in JS/JSX. Stylesheets are deliberately excluded --
// the design system's own nocturne.css uses raw hex and px, so the rule is about
// keeping style values out of components, not out of CSS.

import { readFileSync } from 'node:fs'
import { readdir } from 'node:fs/promises'
import { extname, join, relative, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const SOURCE_DIRS = ['src']
const EXTENSIONS = new Set(['.js', '.jsx'])

const RULES = [
  {
    name: 'raw-hex-color',
    pattern: /#[0-9a-fA-F]{3,8}\b/,
    message: 'Raw hex color - use a design-system color token via var().',
  },
  {
    name: 'raw-px-value',
    pattern: /\b\d+px\b/,
    message: 'Raw px value - use a design-system spacing token via var().',
  },
  {
    name: 'non-system-font',
    // (?=\S) stops \s* from backtracking over the space and defeating the
    // lookahead, which is what the shipped selector's regex does.
    pattern: /font-family\s*:\s*(?=\S)(?!['"]?Inter\b)/i,
    message: 'Font not provided by the design system. Available: Inter.',
  },
]

// Strip comments first so a commented-out value is not reported.
const stripComments = (source) =>
  source.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1')

const LITERAL = /'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`/g

async function* walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) yield* walk(path)
    else if (EXTENSIONS.has(extname(entry.name))) yield path
  }
}

const findings = []
for (const directory of SOURCE_DIRS) {
  for await (const path of walk(join(ROOT, directory))) {
    const source = stripComments(readFileSync(path, 'utf8'))
    for (const match of source.matchAll(LITERAL)) {
      for (const rule of RULES) {
        if (!rule.pattern.test(match[0])) continue
        const line = source.slice(0, match.index).split('\n').length
        findings.push({
          file: relative(ROOT, path),
          line,
          rule: rule.name,
          message: rule.message,
          literal: match[0].slice(0, 60),
        })
      }
    }
  }
}

for (const finding of findings) {
  console.error(`  ${finding.file}:${finding.line}  ${finding.rule}`)
  console.error(`    ${finding.message}`)
  console.error(`    ${finding.literal}`)
}
console.log(
  findings.length === 0
    ? 'design tokens: no adherence violations'
    : `design tokens: ${findings.length} adherence violation(s)`,
)
process.exit(findings.length === 0 ? 0 : 1)
