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
| Token | One item in a modelled or tokenized sequence, such as a word, sentence-boundary marker, initial character, end-of-word marker, or string created by a merge. |
| Vocabulary | The finite set of tokens available to a model or tokenizer, including next-token choices in a language model or initial and learned tokens in BPE. |
| N-gram | A contiguous sequence of `n` tokens used by a count-based language model; the preceding `n - 1` tokens condition the next-token probability. |
| Language model | A model that assigns probabilities to token sequences or to a next token given its preceding tokens. |
| Perplexity | The exponential of the mean negative natural-log probability assigned to evaluation events; lower values indicate better probability assignment on the same evaluation contract. |
| Add-alpha smoothing | A probability estimate that adds the same positive alpha to every possible next-token count and normalizes each conditional distribution over the vocabulary. |
| Unknown token | The `<unk>` token that represents words excluded from the training vocabulary or unseen in training data. |
| Sentence-boundary token | A special token that marks sentence context or termination: `<bos>` supplies left context and `<eos>` is a predictable end event. |
| Byte-pair encoding (BPE) | A subword-tokenization algorithm that repeatedly replaces a selected adjacent token pair with a new token and records an ordered merge rule. This pipeline starts with character tokens. |
| Merge rule | An ordered adjacent token pair, its replacement token, and its learning rank. |
| End-of-word marker | The special token `</w>` appended to each word to preserve word boundaries during training and tokenization. |
| Training corpus | The configured ordered words and positive frequencies used to learn BPE merge rules. |
