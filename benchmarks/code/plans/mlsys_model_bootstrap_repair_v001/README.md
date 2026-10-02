# Local preparation repair: missing ensurepip

The main-v5e `MODEL-tools` execution failed before installing any model package.
The original `benchmark_mlsys_prepare_v001.py` called ordinary `python -m venv`,
which in turn called `ensurepip`. The read-only inspection in
`runs/20260921T205903Z-inspect-model-bootstrap-v001-61854c/execution.log`
confirmed Python 3.13.15 has no `ensurepip`, while system pip 24.1.2 works.
The partial target also has neither pip nor ensurepip.

The new main-campaign preparation path had omitted the workaround already used
by `tools/install_model_tools_v003.py`. The new version reuses its pure command
builder, exact package pins, and isolation probes, without assuming that an old
allocation's package-inventory file exists on the current machine.

## Repair

- Preserve the failed `cpu-tools` directory and all original executed sources.
- Create a fresh `cpu-tools-v002` using `python -m venv --without-pip`.
- Use the existing system pip with `--isolated --python <new-environment>/bin/python`
  for every package operation. Do not upgrade or install into system Python.
- Keep the CPU-only PyTorch and model-tool pins unchanged, including protobuf.
- Check target-prefix isolation, package imports, dependency consistency, and
  global package versions before and after installation.
- Run subsequent input preparation with the new environment's interpreter.
- Preserve command logs, installation reports, and failures in a new archive.

This is pip's documented support for managing an environment that lacks pip:
https://pip.pypa.io/en/stable/topics/python-option/

## Validation and deployment boundary

The user explicitly requested local work only and no further TPU access while
the main measurements are running. The repair source and offline validation are
prepared locally. No new installer or model run is deployed by this repair task;
the active controller, frozen campaign plan, and TPU environment are unchanged.

`tools/check_mlsys_bootstrap_v001.py` checks command targets and preserved paths,
simulates unavailable ensurepip, creates an actual pip-less local environment,
installs a tiny locally generated wheel through external pip, checks its import
isolation, and checks global versions. It uses no network or remote runtime.
The full pinned Linux model-tool installation and real-model runs still require
a separately archived execution after the active measurements are finished.

Old failed and blocked dashboard entries remain historical evidence. A local
repair does not turn those entries into completed model experiments.
