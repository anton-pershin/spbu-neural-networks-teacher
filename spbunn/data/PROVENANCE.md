# Alice corpus provenance

## Work

- Title: *Alice's Adventures in Wonderland*
- Author: Lewis Carroll
- Publication year in the NLTK source header: 1865
- Vendored resource: `carroll-alice.txt`
- Vendored file size: 144,395 bytes
- Vendored file SHA-256: `5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266`

The text is vendored unchanged. The pipeline removes only the first bracketed title line as a documented preprocessing step; the source file itself is not modified.

## Acquisition source

The file came from the official NLTK data distribution, not from a runtime NLTK installation:

- Corpus package: Project Gutenberg Selections (`gutenberg`)
- Archive member: `gutenberg/carroll-alice.txt`
- NLTK data commit: `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a`
- Pinned archive: `https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/gutenberg.zip`
- Archive size: 4,251,829 bytes
- Archive SHA-256: `2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611`
- Retrieval date: 2026-11-18

The original `gutenberg/README` is preserved as `nltk-gutenberg-README.txt`. The original package metadata is preserved as `nltk-gutenberg.xml`; it declares both the license and copyright as `public domain`.

## Reproduction and updates

Use the pinned archive URL above, verify its SHA-256, and extract only `gutenberg/carroll-alice.txt` and `gutenberg/README`. Verify the extracted Alice byte count and SHA-256. Fetch `gutenberg.xml` from the same pinned commit. Do not replace these resources from a mutable branch or an unverified local `nltk_data` directory.

An update requires a deliberate provenance revision, new hashes, and a rerun of preprocessing and numerical regression checks. Normal pipeline runs and CI use only the packaged file and never download this archive.
