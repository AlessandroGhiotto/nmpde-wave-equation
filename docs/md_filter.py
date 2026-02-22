#!/usr/bin/env python3
"""
Doxygen input filter for Markdown files.

Transforms bare <div align="center"> / </div> wrappers that surround Markdown
tables into Doxygen \\htmlonly / \\endhtmlonly blocks.  This lets Doxygen keep
processing the inline Markdown inside table cells (bold, math, etc.) while
still emitting the <div> tag as raw HTML so that the custom CSS class
"centered-table" can centre the table visually.
"""
import sys
import re

_OPEN_DIV = re.compile(r'^\s*<div\s+align=["\']center["\']\s*>\s*$', re.IGNORECASE)
_CLOSE_DIV = re.compile(r'^\s*</div\s*>\s*$', re.IGNORECASE)

def filter_md(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_centered_block = False

    for line in lines:
        if _OPEN_DIV.match(line):
            in_centered_block = True
            out.append('\\htmlonly\n<div class="centered-table">\n\\endhtmlonly\n')
            continue

        if in_centered_block and _CLOSE_DIV.match(line):
            in_centered_block = False
            out.append('\\htmlonly\n</div>\n\\endhtmlonly\n')
            continue

        out.append(line)

    return ''.join(out)

if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        content = f.read()
    sys.stdout.write(filter_md(content))
