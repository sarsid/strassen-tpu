"""Select authorized output groups without deleting historical identities."""
from llm_queue_state_v002 import remaining as all_remaining


def active_dtypes(job):
    values = job.get('active_output_dtypes', list(job['groups']))
    if not values or len(set(values)) != len(values) or not set(values) <= set(job['groups']):
        raise ValueError('Invalid active output scope')
    return values


def remaining(job, rows):
    # Validate whole groups in the complete ledger before applying scope.
    return [dtype for dtype in all_remaining(job, rows) if dtype in active_dtypes(job)]


def active_ids(state):
    return {ident for job in state['jobs'] for dtype in active_dtypes(job)
            for ident in job['groups'][dtype]}


def measured_count(state, rows):
    return len(active_ids(state).intersection(rows))
