# Simple convolution pipeline

## Purpose

M4 provides the simple convolution experiment as a Hydra-configured teaching pipeline. Its image and kernel producer, numerical evaluator, and report builder communicate through explicit in-memory values. Only the shared `ReportWriter` writes the HTML report and figures. The orchestrator closes all Matplotlib figures after output succeeds or fails.

## Input images

The pipeline needs two input images:

- `images.rgb` selects the image for the horizontal and vertical experiments. The reader selects RGB mode, so an alpha channel is discarded.
- `images.grayscale` selects the image for the diagonal experiment. The reader selects one-channel grayscale mode.

The production defaults retain the legacy names `climate_change.png` and `waves.png`. Both supplied PNG files are stored at the repository root. From the source-checkout root, run the pipeline with these defaults:

```bash
python -m spbunn.scripts.simple_convolution
```

To use other images, set either or both paths on the command line or in a configuration overlay:

```bash
python -m spbunn.scripts.simple_convolution \
  images.rgb=/path/to/rgb.png \
  images.grayscale=/path/to/grayscale.png
```

Absolute paths are supported. Relative paths, including the defaults, are resolved from the directory in which the command was originally invoked, including when Hydra changes its job directory. To select an explicit report directory, set both common output values:

```bash
python -m spbunn.scripts.simple_convolution \
  images.rgb=/path/to/climate_change.png \
  images.grayscale=/path/to/waves.png \
  result_dir=/tmp/spbunn-convolution hydra.run.dir=/tmp/spbunn-convolution
```

The production configuration is `config/config_simple_convolution.yaml`. Common M1 settings define `result_dir` and Hydra logging.

## Numerical behavior

Both images receive a batch dimension and are converted to `float64`. The RGB image is normalized before either RGB experiment. The grayscale image remains unnormalized.

The production pipeline uses 7-by-7 kernels and 20 repetitions. The horizontal kernel has ones in its center row, the vertical kernel has ones in its center column, and the diagonal kernel is the identity matrix. Each RGB kernel is replicated into three output-channel filters. Grouped convolution with three groups applies one kernel independently to each RGB channel. The diagonal experiment uses one group.

Every repetition uses `torch.nn.functional.conv2d` with unit stride, no bias, and `padding="same"`. PyTorch performs cross-correlation without flipping the kernel. Same padding uses zero-valued contributions beyond the input boundary.

RGB normalization preserves the legacy tensor reduction. For each horizontal index, it calculates the mean and sample standard deviation over the channel and vertical dimensions. It normalizes the initial RGB input and the result after every horizontal or vertical convolution. The grayscale diagonal result is not normalized.

## Report and figures

Each run writes `simple_convolution.html` and three figures directly into `result_dir`:

| Experiment | Figure artifact stem | Layout |
|---|---|---|
| Horizontal RGB convolution | `stripes_convoluted_h` | Two rows of three channel images |
| Vertical RGB convolution | `stripes_convoluted_v` | Two rows of three channel images |
| Diagonal grayscale convolution | `waves_convoluted` | Two rows with one image each |

The first row shows the original image and the second shows the result after the configured repetitions. All channel panels use the legacy reverse-grayscale color map. The report explains the operations with HTML and LaTeX. The embedded MathJax runtime renders mathematics without network access.

The shared writer selects SVG when the serialized SVG size is at most 102,400 bytes and PNG otherwise. It writes exactly one format per figure stem and removes a stale alternative-format artifact on rerun.

## Small test configuration

`config/config_test_simple_convolution.yaml` uses 3-by-3 kernels and two repetitions with the same configurable image defaults. Automated numerical and pipeline tests generate small RGBA and grayscale PNG files in temporary directories and supply those paths explicitly. The production-settings review run uses the repository-default PNG files.

Run all local checks:

```bash
MPLBACKEND=Agg python -m pytest
./run_linters.sh
python -m build
```

The tests compose both configurations, execute the module in a subprocess from a separate working directory, verify numerical results and legacy kernels, inspect report figures and LaTeX content, check artifact placement and format exclusivity, and verify figure cleanup after success and writer failure.

Human review of the rendered report is required for educational acceptance. Open it with network access disabled to verify LaTeX rendering and inspect all original and convolved panels.
