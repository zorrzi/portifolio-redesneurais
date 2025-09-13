window.MathJax = {
  tex: {inlineMath: [['$', '$'], ['\\(', '\\)']], displayMath: [['$$','$$'], ['\\[','\\]']]},
  options: {skipHtmlTags: ['script','noscript','style','textarea','pre','code']}
};
document$.subscribe(() => { MathJax.typesetPromise(); });
