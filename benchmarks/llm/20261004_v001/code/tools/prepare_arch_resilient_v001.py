"""Freeze an architecture-specific continuation from verified whole comparisons."""
import argparse
import json
from pathlib import Path
import prepare_llm_tradeoff_v002 as campaign
from prepare_arch_v6e_v002 import build as v6plan
from prepare_arch_v5e_v001 import build as v5plan
from arch_resilience_v001 import check_contract, missing


def build(args):
    check_contract(campaign.ROOT)
    remaining = missing(json.loads(args.history.read_text()), args.hardware)
    plan = (v6plan if args.hardware == 'v6e' else v5plan)(args)
    allowed = {u + '-' + part for u in remaining for part in ('screen', 'confirm')}
    allowed |= ({'arch-smoke'} if args.hardware == 'v6e' else {'fp32-smoke', 'bf16-smoke'})
    allowed |= {'export-' + phase for phase in list(allowed)}
    # The supervisor reports/releases after all attempts have been reconciled.
    plan['stages'] = [s for s in plan['stages'] if s['id'] in allowed]
    ids = {s['id'] for s in plan['stages']}
    for stage in plan['stages']:
        stage['requires'] = [d for d in stage.get('requires', []) if d in ids]
    plan.update(transport='tools/run_phase_v007.py',
        title=f'{args.hardware}: automatically supervised continuation ({len(remaining)} remaining comparison units)',
        history=str(args.history.resolve()), automatic_recovery=True,
        decisions=plan['decisions'] + [
            'Only whole screen/confirmation comparisons are reused; incomplete units restart after runtime loss.',
            'The supervisor owns allocation lifetime; successful controller completion does not release this TPU.',
            'All completed phases, failed attempts and allocation identities remain archived.'])
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['freeze', 'launch'])
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--history', type=Path)
    p.add_argument('--hardware', choices=['v6e', 'v5e'])
    p.add_argument('--port', type=int, default=8790)
    for key in ('session', 'endpoint', 'identity', 'controller-python', 'analysis-python'):
        p.add_argument('--' + key)
    args = p.parse_args()
    campaign.build_plan = build
    if args.action == 'freeze':
        campaign.freeze(args)
    else:
        raise ValueError('The supervisor launches its own controller without another dashboard')


if __name__ == '__main__':
    main()
