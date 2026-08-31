"""Disjoint-text validation of the calibrated terminal last-4 policy."""

from pathlib import Path

import benchmark_checkpoint_schedules as schedules


schedules.OUTPUT = Path(
    "/content/results/strassen_checkpoint_holdout.jsonl"
)
schedules.CALIBRATION_SOURCE_SHA256 = (
    "cc78c842b275460c36b715d610444bd889024c748d61f182d08045ec53daaf07"
)
schedules.SCHEDULES = {
    "last_4": {"both": frozenset(range(28, 32)), "up": frozenset()},
}
schedules.layer_bench.TEXTS = (
    "Rain moved across the valley in silver bands while the old stone bridge "
    "held its place above the river and the road disappeared into mist.",
    "Economic measurements summarize many individual choices, so a useful "
    "forecast states its assumptions and gives a range rather than one number.",
    "The telescope gathered faint light for several hours. After calibration, "
    "the image revealed a distant structure that no single exposure could show.",
    "A legal rule can be simple to state yet difficult to apply because facts, "
    "precedent, language, and the purpose of the rule must be considered together.",
    "During rehearsal the actors changed the pace of the scene. A longer pause "
    "made the final line quieter, clearer, and more surprising to the audience.",
    "An algorithm is useful only in a setting: input sizes, hardware, precision, "
    "memory traffic, and surrounding operations decide whether its theory pays.",
    "The recipe began with ordinary ingredients, but temperature and timing "
    "changed their texture. The cook wrote down each adjustment for the next trial.",
    "Maps select details for a purpose. A subway diagram, a weather chart, and a "
    "topographic survey describe the same region through very different questions.",
)


if __name__ == "__main__":
    schedules.main()
