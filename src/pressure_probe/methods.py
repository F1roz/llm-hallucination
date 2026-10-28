"""The detectors compared in the paper: column name, label, access type, role and cost."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    access: str  # "text" needs only generated text; "logits" needs token probabilities
    scope: str  # "all" scored on every question; "matched" only on the matched set
    role: str = "baseline"  # "baseline", "ours", or "variant" (other scores from the same pressure replies)

    @property
    def is_ours(self) -> bool:
        return self.role == "ours"


METHODS = (
    Method("logprob", "Token log probability", "logits", "all"),
    Method("ptrue", "P(True)", "logits", "all"),
    Method("verbal", "Verbalized confidence", "text", "all"),
    Method("semantic_entropy", "Semantic Entropy", "text", "all"),
    Method("selfcheck_nli", "SelfCheckGPT-NLI", "text", "all"),
    Method("cross_family", "Cross family disagreement", "text", "all"),
    Method("metaqa", "MetaQA", "text", "matched"),
    Method("dfr", "Doubt flip rate (ours)", "text", "matched", "ours"),
    Method("dfr_cross", "Doubt flip + cross family (ours)", "text", "matched", "ours"),
    Method("f_counter", "Counterclaim flip rate", "text", "matched", "variant"),
    Method("cps", "Contrastive pressure (CPS)", "text", "matched", "variant"),
)

BASELINES = tuple(m for m in METHODS if m.role == "baseline")
PRIMARY = "dfr"
SAMPLING_BASELINES = ("semantic_entropy", "selfcheck_nli")


def calls_per_question(config: Config) -> dict[str, int]:
    """Generations each method needs per question at deployment, including the answer itself."""
    samples, pressure, mutations = config.n_samples, config.n_pressure, config.metaqa_mutations
    return {
        "logprob": 1,
        "ptrue": 2,
        "verbal": 2,
        "semantic_entropy": 1 + samples,
        "selfcheck_nli": 1 + samples,
        "cross_family": 2,
        "metaqa": 1 + 2 + 2 * mutations,
        "dfr": 1 + pressure,
        "dfr_cross": 1 + pressure + 1,
        "f_counter": 1 + 1 + pressure,
        "cps": 1 + 1 + 2 * pressure,
    }
