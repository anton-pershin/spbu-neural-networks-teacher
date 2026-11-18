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
| N-gram | A contiguous sequence of `n` tokens used by a count-based language model; the preceding `n - 1` tokens condition the next-token probability. |
| Token | One item in a modelled sequence, such as a word or a sentence-boundary marker. |
| Vocabulary | The finite set of tokens that can receive probability mass as the next token. |
| Language model | A model that assigns probabilities to token sequences or to a next token given its preceding tokens. |
| Perplexity | The exponential of the mean negative natural-log probability assigned to evaluation events; lower values indicate better probability assignment on the same evaluation contract. |
| Add-alpha smoothing | A probability estimate that adds the same positive alpha to every possible next-token count and normalizes each conditional distribution over the vocabulary. |
| Unknown token | The `<unk>` token that represents words excluded from the training vocabulary or unseen in training data. |
| Sentence-boundary token | A special token that marks sentence context or termination: `<bos>` supplies left context and `<eos>` is a predictable end event. |
