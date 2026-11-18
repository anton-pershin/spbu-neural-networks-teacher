## Terminology

### General

| Term | Definition |
|------|------------|
| GetSomeMath | The course-content project hosted at `anton-pershin/getsomemath`, into which generated reports and figures are intended to be integrated. |
| Run directory | The Hydra-configured directory in which a pipeline writes its HTML report and all generated figures. |

### Reporting

| Term | Definition |
|------|------------|
| HTML report with LaTeX support | An HTML artifact that renders the pipeline's explanatory content and supports mathematical notation written in LaTeX. |
| Report model | The in-memory representation of a pipeline's output, containing HTML content, figure objects, and artifact names. |
| Report builder | The pipeline-specific component that converts experiment results into a report model. |
| Report writer | The shared component that writes a report model's HTML and figures to the run directory and applies the SVG/PNG size-threshold rule. |

### Sequence modelling

| Term | Definition |
|------|------------|
| Byte-pair encoding (BPE) | A subword-tokenization algorithm that repeatedly replaces a selected adjacent token pair with a new token and records an ordered merge rule. This pipeline starts with character tokens. |
| Token | One item in a tokenized word: an initial character, the end-of-word marker, or a string created by a merge. |
| Vocabulary | The set of initial tokens and tokens added by learned merge rules. |
| Merge rule | An ordered adjacent token pair, its replacement token, and its learning rank. |
| End-of-word marker | The special token `</w>` appended to each word to preserve word boundaries during training and tokenization. |
| Training corpus | The configured ordered words and positive frequencies used to learn BPE merge rules. |
