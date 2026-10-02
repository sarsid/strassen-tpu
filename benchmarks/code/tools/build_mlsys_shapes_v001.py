"""Build frozen development-shape and LLM-shape manifests without old timings."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / 'outputs/01a0bf32-5290-7bb0-af6f-d1a7718dbb47/probe-comparison-v001/joined_by_shape.csv'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def build(output_root):
    prior = json.loads((ROOT / 'configs/gemma_depth_tune_v001/campaign.json').read_text())
    distributions = json.loads((ROOT / 'configs/two_level_tune_v001/distributions.json').read_text())
    depth_tiles = prior['tiles']
    additions = {
        'cubic': [[16,128,128], [128,512,512], [256,1024,512], [512,1024,1024],
                  [1024,512,1024], [1536,1536,512], [1024,1024,1024], [1024,1024,1536]],
        'one_level': [[16,256,256], [128,512,512], [256,1024,512], [512,1024,1024],
                      [1536,1536,512], [768,768,512], [1024,1024,1024], [512,1536,512]],
        'two_level': [[32,512,512], [128,512,512], [256,1024,512], [512,1024,1024],
                      [1536,1536,512], [768,1024,512], [1024,1024,1024], [512,1536,512]],
    }
    families = {'native': [dict(algorithm='native', variant='plain', tile=None, **item)
                            for item in prior['native_candidates']]}
    for family, more in additions.items():
        tiles = depth_tiles + more
        assert len(tiles) == len({tuple(t) for t in tiles}) == 16
        families[family] = [dict(candidate_id=family+'_'+'_'.join(map(str,tile)),
            algorithm='cubic_full' if family == 'cubic' else 'strassen',
            variant={'cubic':'output_accumulator', 'one_level':'interleaved_output_accumulator',
                     'two_level':'two_level_scratch'}[family], tile=tile, compiler_options={}) for tile in tiles]
    grouped = {}
    csv_rows = list(csv.DictReader(CSV.open()))
    for index, row in enumerate(csv_rows, 2):
        key = tuple(int(row[axis]) for axis in ('M', 'K', 'N'))
        item = grouped.setdefault(key, dict(id='shape_m%d_k%d_n%d' % key, m=key[0], k=key[1], n=key[2], source_rows=[]))
        item['source_rows'].append(dict(csv_line=index, probe=row['Probe'], shape_id=row['Shape ID'],
            sampling_group=row['Sampling group'], grid_membership=row['Grid membership']))
    shapes = sorted(grouped.values(), key=lambda s: (s['m']*s['k']*s['n'], s['m'], s['k'], s['n']))
    assert len(csv_rows) == 180 and len(shapes) == 168
    llm_source = ROOT / 'configs/llm_large_shapes_v001/shapes.json'
    llm = json.loads(llm_source.read_text())['shapes']
    assert len(llm) == len({(s['m'],s['k'],s['n']) for s in llm}) == 18
    for campaign_id, inventory, provenance, smoke_ids in [
        ('mlsys_shapes_v001', shapes, dict(source_csv=str(CSV.relative_to(ROOT)),
            source_sha256=hashlib.sha256(CSV.read_bytes()).hexdigest(), source_rows=180, distinct_shapes=168,
            deduplication_axes=['M','K','N'], previous_timings_used=False),
            [shapes[0]['id'], next(s['id'] for s in shapes if (s['m'],s['k'],s['n']) == (8192,8192,8192))]),
        ('mlsys_llm_shapes_v001', llm, dict(source_manifest=str(llm_source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(llm_source.read_bytes()).hexdigest(), distinct_shapes=18,
            scope='Synthetic operands at historical Qwen, Mistral, Gemma dimensions; real execution is a separate stage.'),
            [llm[0]['id']]),
    ]:
        campaign = {key: copy.deepcopy(prior[key]) for key in
                    ('schema_version','device','precision','memory','correctness','execution_contract')}
        campaign.update(campaign_id=campaign_id, shape_manifest='shapes.json', distribution_manifest='distributions.json',
            arms=['native','cubic','one_level','two_level'], candidate_families=copy.deepcopy(families),
            timing={'smoke':{'warmups':1,'repeats':3}, 'screen':{'warmups':3,'repeats':10},
                    'confirm':{'warmups':5,'repeats':30}},
            screen_seed=2026092701, smoke_seed=2026092601,
            confirm_seeds=[2026192701,2026292701,2026392701], order_seed=927510,
            smoke_shape_ids=smoke_ids,
            selection_policy={'top_k':3, 'near_fraction':0.05, 'max_confirm_per_family':4,
                              'bootstrap_replicates':4000,'confidence_level':0.95},
            timing_scope='call_only: device padding, multiplication, crop; host transfer and compilation excluded',
            candidate_design='Sixteen independently searched tiles per custom family. Eight prior depth-probe tiles plus eight prior-informed small, skinny, or historically useful tiles. Native has four registered compiler settings; compiler controls Native tiling. This is a bounded prior-informed search, not an exhaustive optimum.',
            headline_rule='Freeze fastest eligible screening mean separately per family. Confirmation never changes the headline choice. Retain top three plus candidates within five percent, capped at four; record candidates omitted by cap.',
            development_set=True, unseen_shape_validation=False)
        campaign['precision']['reference_dtype'] = 'float64'
        campaign['precision']['note'] = 'All five methods use exact same BF16 operands, DEFAULT dots, FP32 accumulation/output. Strassen pre-additions round in BF16 at each level; reference uses host float64.'
        campaign['correctness'].update(reference='host_numpy_fp64', scope='Full-output finiteness plus full or sampled all-K FP64 numerical metrics; eligibility threshold is not a model-quality guarantee.')
        campaign['memory']['execution_order'] = 'Ascending useful volume, deterministic candidate interleaving; one confirmation group per shape reuses compiled finalists across three fresh input seeds.'
        campaign['memory']['max_full_reference_flops'] = 2000000000
        folder = output_root / campaign_id
        write(folder / 'campaign.json', campaign)
        write(folder / 'shapes.json', dict(schema_version=1, shape_order=['m','k','n'],
            tile_order=['bm','bn','bk'], provenance=provenance, shapes=inventory))
        write(folder / 'distributions.json', distributions)
    return dict(passed=True, shapes=168, llm_shapes=18, candidate_counts={k:len(v)for k,v in families.items()},
                main_screen_outcomes=168*52, llm_screen_outcomes=18*52)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output_root), sort_keys=True))
