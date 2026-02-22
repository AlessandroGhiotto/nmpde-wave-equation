// Extend MathJax 3 configuration to recognise $...$ as inline math.
// This is required because Doxygen does not convert $...$ delimiters
// inside Markdown table cells to \(...\), so MathJax must handle them.
// This file is loaded by Doxygen's MATHJAX_CODEFILE after window.MathJax
// is defined but before the MathJax script is fetched and executed.
window.MathJax = window.MathJax || {};
window.MathJax.tex = window.MathJax.tex || {};
window.MathJax.tex.inlineMath = [['$', '$'], ['\\(', '\\)']];