"""Four-model scheduling with an explicit barrier for unqualified architectures."""
import copy
from llm_queue_state_v002 import identity, plan
from llm_scope_v001 import active_ids

ORDER = ['qwen3_8b', 'qwen3_32b', 'gemma4_31b', 'qwen3_14b']
REASON = ('Gemma4-31B requires a qualified Gemma4 adapter and official-model '
          'oracle: local/global head geometry, shared global K/V and partial RoPE '
          'are not supported by the current Gemma3 adapter.')


def revise(state, protocol, metadata):
    result = copy.deepcopy(state)
    assert metadata['id'] == 'gemma4_31b' and metadata['config']['model_type'] == 'gemma4'
    removed = [j for j in result['jobs'] if j['model'] not in ORDER]
    kept = [j for j in result['jobs'] if j['model'] in ORDER]
    if not any(j['model'] == 'gemma4_31b' for j in kept):
        gemma_protocol = dict(protocol, models=[metadata], output_dtypes=['bfloat16'])
        for job in plan(gemma_protocol):
            job.update(status='awaiting_qualification', active_output_dtypes=['bfloat16'],
                       qualification_status='awaiting_qualification', reason=REASON)
            kept.append(job)
    # Stable within-model order preserves the existing workload order and attempts.
    kept.sort(key=lambda j: ORDER.index(j['model']))
    session_ids = set(result.get('active_session', {}).get('job_ids', []))
    if not session_ids <= {j['job_id'] for j in kept}:
        raise ValueError('Cannot defer a workload in the active remote session')
    result['jobs'] = kept
    previous = result.setdefault('deferred_jobs', [])
    known = {j['job_id'] for j in previous}
    previous.extend(j for j in removed if j['job_id'] not in known)
    result.update(model_order=ORDER.copy(), active_output_dtypes=['bfloat16'],
                  scheduling='model-major', scheduling_note='Finish each model in the requested order; qualify Gemma4 before it runs.',
                  scope_revision='four_models_20261003_v001',
                  active_plan='planned_grid_four_models_v001/planned_entries.jsonl',
                  additional_model_metadata='inputs_gemma4_v001/models.json',
                  orchestration_version='model-session-four-models-v001')
    result['expected_comparisons'] = len(active_ids(result))
    assert len(kept) == 40 and result['expected_comparisons'] == 200
    return result


def pending_job(state):
    # Do not silently skip a requested model because its adapter is not yet ready.
    return next((j for j in state['jobs'] if j['status'] in
                 ('queued', 'awaiting_qualification')), None)


def qualification_block(job):
    return job and (job['status'] == 'awaiting_qualification' or
                    job.get('qualification_status') == 'awaiting_qualification')


def planned_entries(state, previous, protocol):
    old = {r['entry_id']: r for r in previous}
    rows = []
    for job in state['jobs']:
        for algorithm, ident in zip(protocol['algorithms'], job['groups']['bfloat16']):
            key = {k: job[k] for k in ('model', 'revision', 'batch', 'sequence', 'architecture')}
            key.update(algorithm=algorithm, output_dtype='bfloat16')
            assert identity(key) == ident
            rows.append(old.get(ident) or dict(key, entry_id=ident, model_type='gemma4_text',
                        status='planned', measurement=None, projection_shapes_mkn=None,
                        requires=['gemma4_adapter', 'official_checkpoint_qualification',
                                  'frozen_profile', 'independent_confirmation']))
    return rows
