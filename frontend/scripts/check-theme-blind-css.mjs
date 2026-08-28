#!/usr/bin/env node
// Flags CSS rules that paint a light background with a raw hex and have no
// [data-theme='dark'] override, which renders as a light island on a dark page.
//
// The stylesheet still holds many raw hex values by design; only this specific
// combination is a bug, so that is all this checks.

import { readFileSync } from 'node:fs'
import { dirname, join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const FILES = ['src/App.css', 'src/index.css']
const LIGHTNESS_LIMIT = 0.75

const luminance = (hex) => {
  let value = hex.slice(1)
  if (value.length === 3) value = [...value].map((c) => c + c).join('')
  if (value.length !== 6) return null
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(value.slice(i, i + 2), 16) / 255)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

const findings = []
for (const file of FILES) {
  const source = readFileSync(join(ROOT, file), 'utf8')

  const overridden = new Set()
  for (const [, selector] of source.matchAll(/\[data-theme='dark'\]([^{]*)\{[^}]*\}/g)) {
    for (const [, name] of selector.matchAll(/\.([a-z-]+)/g)) overridden.add(name)
  }

  for (const rule of source.matchAll(/([^{}]+)\{([^}]*)\}/g)) {
    const selector = rule[1].trim()
    if (selector.includes('data-theme') || selector.startsWith('@') || selector.startsWith(':root')) continue
    const classes = [...selector.matchAll(/\.([a-z-]+)/g)].map((m) => m[1])
    if (classes.some((name) => overridden.has(name))) continue

    for (const [, , declared] of rule[2].matchAll(/(background[a-z-]*)\s*:\s*([^;]+)/g)) {
      // A hex inside var(--token, #fallback) is themed by the token, not blind.
      const value = declared.replace(/var\([^)]*\)/g, ' ')
      for (const [hex] of value.matchAll(/#[0-9a-fA-F]{3,6}\b/g)) {
        const light = luminance(hex)
        if (light !== null && light > LIGHTNESS_LIMIT) {
          const line = source.slice(0, rule.index).split('\n').length
          findings.push({ file, line, selector: selector.slice(0, 60), hex })
        }
      }
    }
  }
}

for (const finding of findings) {
  console.error(`  ${relative('.', finding.file)}:${finding.line}  ${finding.selector}`)
  console.error(`    Light background ${finding.hex} with no [data-theme='dark'] override.`)
  console.error('    Use a token that both themes define, or add a dark override.')
}
console.log(findings.length === 0
  ? 'theme-blind css: none'
  : `theme-blind css: ${findings.length} rule(s)`)
process.exit(findings.length === 0 ? 0 : 1)
