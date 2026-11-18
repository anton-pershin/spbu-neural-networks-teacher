"""Corpus, configuration, production, and executable n-gram pipeline checks."""

import os
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from omegaconf import DictConfig

from spbunn.ngram.pipeline import (
    build_training_vocabulary,
    load_packaged_corpus,
    produce_ngram_input,
    split_sentences,
    tokenize_alice_corpus,
)
from spbunn.reporting import ReportWriter
from spbunn.scripts.ngram import ngram
from spbunn.utils.common import get_project_path
from tests.ngram_helpers import compose_ngram_config, production_ngram_result

ALICE_SHA256 = "5ce9042c0d8536b072b61d4c7c4e6d70c72d9dbb5f3d816261326d9280749266"


def test_packaged_corpus_hash_and_provenance_are_available() -> None:
    text = load_packaged_corpus("carroll-alice.txt", ALICE_SHA256)

    assert text.startswith("[Alice's Adventures in Wonderland by Lewis Carroll 1865]\n")
    assert len(text.encode("utf-8")) == 144395
    package_files = files("spbunn.data")
    for resource in (
        "PROVENANCE.md",
        "nltk-gutenberg-README.txt",
        "nltk-gutenberg.xml",
    ):
        assert package_files.joinpath(resource).is_file()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_packaged_corpus("carroll-alice.txt", "0" * 64)
    with pytest.raises(ValueError, match="without a path"):
        load_packaged_corpus("../carroll-alice.txt", ALICE_SHA256)


def test_tokenizer_protects_abbreviations_and_drops_short_sentences() -> None:
    text = (
        "[Alice's Adventures in Wonderland by Lewis Carroll 1865]\n\n"
        "Mr. Rabbit came home. Dr. Alice left! two words?"
    )

    sentences = tokenize_alice_corpus(text)

    assert sentences == (
        ("mr", "rabbit", "came", "home"),
        ("dr", "alice", "left"),
    )


def test_tokenizer_rejects_unexpected_header() -> None:
    with pytest.raises(ValueError, match="Unexpected NLTK Alice title"):
        tokenize_alice_corpus("[Different title]\nA valid sentence here.")


def test_sentence_split_is_deterministic_and_disjoint() -> None:
    sentences = tuple((f"word{index}", "middle", "end") for index in range(20))

    first = split_sentences(
        sentences, seed=9, training_fraction=0.5, validation_fraction=0.25
    )
    second = split_sentences(
        sentences, seed=9, training_fraction=0.5, validation_fraction=0.25
    )

    assert first == second
    assert tuple(map(len, first)) == (10, 5, 5)
    training, validation, test = (set(split) for split in first)
    assert training.isdisjoint(validation)
    assert training.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert set.union(training, validation, test) == set(sentences)
    with pytest.raises(ValueError, match="must not be empty"):
        split_sentences(
            sentences[:2], seed=9, training_fraction=0.5, validation_fraction=0.25
        )


def test_training_vocabulary_uses_only_training_counts() -> None:
    training = (("a", "b", "a"), ("b", "c"))

    assert build_training_vocabulary(training, 2) == frozenset({"a", "b"})


def test_production_corpus_and_model_results_match_investigation() -> None:
    experiment = produce_ngram_input(compose_ngram_config("config_ngram"))
    result = production_ngram_result()

    assert experiment.sentence_count == 1503
    assert tuple(
        map(
            len,
            (
                experiment.training_sentences,
                experiment.validation_sentences,
                experiment.test_sentences,
            ),
        )
    ) == (1052, 226, 225)
    assert len(experiment.prediction_vocabulary) == 1163
    assert result.corpus.training_word_count == 18185
    assert result.corpus.validation_word_count == 4263
    assert result.corpus.test_word_count == 3852

    assert [item.selected_alpha for item in result.order_results] == [0.001, 0.03, 0.01]
    assert [item.observed_ngram_type_count for item in result.order_results] == [
        1163,
        9315,
        15336,
    ]
    assert [item.observed_context_count for item in result.order_results] == [
        1,
        1163,
        8905,
    ]
    assert [
        item.validation_metrics.perplexity for item in result.order_results
    ] == pytest.approx([189.1783163, 123.1952975, 349.1493808])
    assert [
        item.training_metrics.perplexity for item in result.order_results
    ] == pytest.approx([238.8346696, 40.8957310, 14.7827106])
    assert [
        item.test_metrics.perplexity for item in result.order_results
    ] == pytest.approx([201.8315340, 125.2136049, 348.8330133])
    assert [item.test_metrics.event_count for item in result.order_results] == [
        4077,
        4077,
        4077,
    ]
    assert result.order_results[1].test_metrics.unseen_ngram_fraction == pytest.approx(
        0.2852587687
    )
    assert result.order_results[2].test_metrics.unseen_ngram_fraction == pytest.approx(
        0.6661761099
    )
    assert result.order_results[
        2
    ].test_metrics.unseen_context_fraction == pytest.approx(0.2752023547)


def test_production_top_k_and_samples_are_exact_regressions() -> None:
    result = production_ngram_result()

    bigram = result.probability_distributions[1]
    assert bigram.context == ("the",)
    assert [(item.token, item.count) for item in bigram.items] == [
        ("<unk>", 113),
        ("queen", 44),
        ("king", 43),
        ("gryphon", 39),
        ("mock", 38),
    ]
    assert [item.probability for item in bigram.items] == pytest.approx(
        [0.09863949, 0.03842428, 0.03755160, 0.03406086, 0.03318818]
    )
    for distribution in result.probability_distributions:
        assert sum(
            item.probability for item in distribution.items
        ) + distribution.remaining_probability == pytest.approx(1.0)

    samples = {
        (sample.order, sample.seed): sample for sample in result.generated_samples
    }
    assert len(samples) == 9
    assert (
        " ".join(samples[2, 42].tokens)
        == "said alice because four sense repeated turning call it"
    )
    assert samples[2, 42].stop_reason == "eos"
    assert not samples[1, 123].tokens
    assert samples[3, 7].stop_reason == "length_limit"
    assert len(samples[3, 7].tokens) == 30


@pytest.mark.parametrize("config_name", ["config_ngram", "config_test_ngram"])
def test_configs_compose_common_run_directory(config_name: str) -> None:
    cfg = compose_ngram_config(config_name)

    assert isinstance(cfg, DictConfig)
    assert cfg.result_dir == cfg.hydra_dir
    assert cfg.hydra_root
    assert cfg.report.artifact_name.endswith(".html")
    assert cfg.top_k == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["orders=[]"], "non-empty and positive"),
        (["orders=[1,1]"], "must be unique"),
        (["orders=[0]"], "non-empty and positive"),
        (["orders=[1.5]"], "finite integer"),
        (["smoothing_alphas=[]"], "finite positive"),
        (["smoothing_alphas=[0.1,0.1]"], "must contain unique"),
        (["smoothing_alphas=[0.0]"], "finite positive"),
        (["split.seed=-1"], "split seed"),
        (["split.seed=-0.5"], "finite integer"),
        (
            ["split.training_fraction=0.8", "split.validation_fraction=0.3"],
            "between zero and one",
        ),
        (["minimum_token_count=0"], "minimum token count"),
        (["minimum_token_count=1.5"], "finite integer"),
        (["top_k=0"], "top-k"),
        (["top_k=5.9"], "finite integer"),
        (["generation.seeds=[]"], "non-empty and non-negative"),
        (["generation.seeds=[-1]"], "non-empty and non-negative"),
        (["generation.seeds=[-0.5]"], "finite integer"),
        (["generation.max_tokens=0"], "token limit"),
        (["generation.max_tokens=30.9"], "finite integer"),
        (["probability_queries=[]"], "At least one probability query"),
        (["probability_queries=[{order:4,context:[]}]"], "unconfigured order"),
        (["probability_queries=[{order:2,context:[of,the]}]"], "requires 1 context"),
    ],
)
def test_invalid_pipeline_settings_are_rejected(
    overrides: list[str], message: str
) -> None:
    cfg = compose_ngram_config(overrides=overrides)

    with pytest.raises(ValueError, match=message):
        produce_ngram_input(cfg)


def test_executable_pipeline_writes_complete_offline_report(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    working_directory = tmp_path / "cwd"
    working_directory.mkdir()
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(get_project_path()), environment.get("PYTHONPATH", "")])
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "spbunn.scripts.ngram",
            "--config-name",
            "config_test_ngram",
            f"hydra.run.dir={run_directory}",
            f"result_dir={run_directory}",
            "hydra.job.chdir=False",
        ],
        cwd=working_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    report_path = run_directory / "test_ngram.html"
    html = report_path.read_text(encoding="utf-8")
    assert 'id="mathjax-runtime"' in html
    assert re.search(r"<script\b[^>]*\bsrc=", html) is None
    assert "Top-k conditional probability distributions" in html
    assert "&lt;unk&gt;" in html
    assert "<unk>" not in html
    image_names = re.findall(r'<img src="([^"]+)"', html)
    assert len(image_names) == 4
    assert {Path(name).stem for name in image_names} == {
        "ngram_validation_perplexity",
        "ngram_train_test_perplexity",
        "ngram_test_sparsity",
        "ngram_top_k_probabilities",
    }
    for image_name in image_names:
        image_path = run_directory / image_name
        assert image_path.is_file()
        assert image_path.suffix in {".svg", ".png"}
        alternative = ".png" if image_path.suffix == ".svg" else ".svg"
        assert not image_path.with_suffix(alternative).exists()
    assert not list(working_directory.glob("*.svg"))
    assert not list(working_directory.glob("*.png"))


def test_orchestrator_closes_figures_after_success(tmp_path: Path) -> None:
    cfg = compose_ngram_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    ngram(cfg)

    assert not plt.get_fignums()


def test_orchestrator_closes_figures_after_writer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = compose_ngram_config(overrides=[f"result_dir={tmp_path / 'run'}"])

    def fail_write(self: ReportWriter, report: object) -> Path:
        del self, report
        raise RuntimeError("controlled writer failure")

    monkeypatch.setattr(ReportWriter, "write", fail_write)
    with pytest.raises(RuntimeError, match="controlled writer failure"):
        ngram(cfg)
    assert not plt.get_fignums()
