document.querySelectorAll('pre code').forEach(block => {
  const lines = block.textContent.split('\n');

  // remove empty first/last lines
  while (lines.length && lines[0].trim() === '') lines.shift();
  while (lines.length && lines[lines.length - 1].trim() === '') lines.pop();

  // find smallest indentation
  const indent = Math.min(
    ...lines
      .filter(line => line.trim())
      .map(line => line.match(/^(\s*)/)[0].length)
  );

  // remove that indentation
  block.textContent = lines.map(line => line.slice(indent)).join('\n');
});