"""Scan public checkpoint payloads, including nested archives, without printing secrets.

This is a targeted credential-pattern check, not a proof that arbitrary data
contains no secrets. Only relative file/member paths and rule names are emitted.
"""
import argparse
import gzip
import io
import json
from pathlib import Path
import re
import tarfile


RULES = {
    'known_token_prefix': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,}|ya29\.[A-Za-z0-9_.-]{25,}|AIza[0-9A-Za-z_-]{30,})'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'jwt': re.compile(rb'eyJ[A-Za-z0-9_-]{18,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}'),
    'credential_json_value': re.compile(rb'"(?:access_token|refresh_token|id_token|runtime_proxy_token|proxy_token|password|client_secret|authorization)"\s*:\s*"[^"\n\\]{16,}"', re.I),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    findings = []
    counts = {'payloads': 0, 'uncompressed_bytes': 0}

    def scan_stream(stream, name):
        counts['payloads'] += 1
        tail = b''
        found = set()
        while True:
            chunk = stream.read(4 * 1024**2)
            if not chunk:
                break
            counts['uncompressed_bytes'] += len(chunk)
            data = tail + chunk
            for label, pattern in RULES.items():
                if label not in found and pattern.search(data):
                    findings.append(dict(path=name, rule=label))
                    found.add(label)
            tail = data[-2048:]

    def visit(stream, name, depth=0):
        assert depth <= 8, name
        if name.endswith(('.tar.gz', '.tgz', '.tar')):
            with tarfile.open(fileobj=stream, mode='r|*') as archive:
                for member in archive:
                    if member.isfile():
                        with archive.extractfile(member) as child:
                            visit(child, name + '::' + member.name, depth + 1)
        elif name.endswith('.gz'):
            with gzip.GzipFile(fileobj=stream) as raw:
                scan_stream(raw, name)
        else:
            scan_stream(stream, name)

    root = args.root.resolve()
    for path in sorted(root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            assert not path.is_symlink(), str(path.relative_to(root))
            with path.open('rb') as stream:
                visit(stream, str(path.relative_to(root)))
    result = dict(passed=not findings, findings=findings, **counts,
                  limitation='Targeted known credential formats and JSON fields; not a general secret-proof claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
