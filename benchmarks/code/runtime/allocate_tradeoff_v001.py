"""One explicit v6e allocation attempt with sanitized service error details."""
import json
import colab_control_v003 as control

if __name__ == '__main__':
    try:
        raise SystemExit(control.main())
    except Exception as exc:
        record = {'kind': 'allocation_error', **control.safe_error(exc)}
        try:
            body = getattr(exc, 'response_body', '').removeprefix(")]}'\n")
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                record['service_error'] = {k: v for k, v in parsed.items()
                    if k in ('error', 'message', 'reason', 'code', 'status')}
        except ValueError:
            record['body_format'] = 'non-json'
        control.emit(record)
        raise SystemExit(1)
