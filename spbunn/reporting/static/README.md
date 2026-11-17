# Vendored MathJax resource

`tex-svg-full.js` is the unmodified MathJax 3.2.2 full TeX-to-SVG browser component. The report writer embeds it directly in each HTML report. The full TeX component avoids dynamic TeX-extension downloads, and SVG output stores glyph paths instead of requesting web fonts. The writer disables the optional MathJax menu to prevent optional accessibility-component downloads.

## Provenance

- Upstream package: `mathjax` 3.2.2
- Source archive: `https://registry.npmjs.org/mathjax/-/mathjax-3.2.2.tgz`
- Source archive SHA-256: `1b9c0a1c44df864e915690558e72adb9cc5203360daefd385084ced3b6c64c09`
- Source file in the archive: `package/es5/tex-svg-full.js`
- Vendored file SHA-256: `a4354ff94fd868aea0cc6eaaa79a57fda0588646fc46ee3700a349ee0a11cbe6`
- License and notices: `MathJax-LICENSE`, copied from `package/LICENSE` in the same archive

The JavaScript file has no local modifications. When updating it, record the new version and checksums here, copy the corresponding license, run the reporting tests, and inspect the generated report with network access disabled.
