#!/usr/bin/env python3
"""Fetch two public Qwen3 BF16 weight tensors using strict bounded HTTP ranges.

No authentication or Hugging Face token is used. Resolve a public repository
revision first, then fetch config, license, headers and tensors at that exact
commit. A server returning HTTP 200 to a range request is refused before its
body is read. Output directories and evidence files are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import uuid

import requests

MODEL_ID = 'Qwen/Qwen3-0.6B'
HF_BASE = 'https://huggingface.co'
TARGETS = (
    {'name': 'model.layers.0.self_attn.q_proj.weight',
     'shape': [2048, 1024], 'path': 'tensors/layer0_q_proj.bf16'},
    {'name': 'model.layers.0.mlp.down_proj.weight',
     'shape': [1024, 3072], 'path': 'tensors/layer0_down_proj.bf16'},
)
MAX_HEADER_BYTES = 16 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024


class FetchValidationError(RuntimeError):
    """A validation message authored locally, without response URLs or secrets."""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def strict_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FetchValidationError('JSON contains a duplicate object key')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_pairs)


def public_filename(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise FetchValidationError('Repository metadata contains an unsupported filename')
    return value


def resolved_url(revision, filename):
    return HF_BASE + '/' + MODEL_ID + '/resolve/' + revision + '/' + public_filename(filename)


class PublicFetcher:
    def __init__(self, output):
        self.output = output
        self.requests_dir = output / 'requests'
        self.requests_dir.mkdir()
        self.index = 0
        self.session = requests.Session()
        # Disable .netrc, proxy credentials and environment-derived auth.
        self.session.trust_env = False
        self.session.auth = None
        self.session.headers.update({'Accept-Encoding': 'identity',
                                     'User-Agent': 'Strassen-MM-Focus-public-weight-fetch/1'})

    def close(self):
        self.session.close()

    def fetch(self, url, destination, *, limit, byte_range=None, expected_total=None):
        self.index += 1
        record = {'index': self.index, 'started_utc': utc_now(), 'canonical_url': url,
                  'output_path': str(destination.relative_to(self.output)),
                  'authentication': 'anonymous; no token, auth header or netrc',
                  'maximum_body_bytes': limit, 'range': list(byte_range) if byte_range else None,
                  'complete': False, 'bytes_written': 0}
        response = None
        try:
            headers = {}
            params = None
            expected_bytes = None
            if byte_range is not None:
                first, last = byte_range
                if first < 0 or last < first:
                    raise FetchValidationError('Invalid requested byte range')
                expected_bytes = last - first + 1
                if expected_bytes > limit:
                    raise FetchValidationError('Requested range exceeds the fixed byte limit')
                headers['Range'] = f'bytes={first}-{last}'
                # Some public CDN caches do not vary their cache key by Range.
                # A fresh public nonce avoids reuse of a different range response.
                params = {'download': 'true', 'strassen_range': uuid.uuid4().hex}
            self.session.cookies.clear()
            response = self.session.get(url, headers=headers, params=params, stream=True,
                                        allow_redirects=True, timeout=(30, 120))
            record['http_status'] = response.status_code
            record['content_range'] = response.headers.get('Content-Range')
            record['content_length'] = response.headers.get('Content-Length')
            record['content_encoding'] = response.headers.get('Content-Encoding')
            # Never save response.url, redirect Location, cookies, auth headers,
            # exception text from requests, or CDN signed query credentials.
            if byte_range is not None:
                if response.status_code != 206:
                    raise FetchValidationError('Range GET must return HTTP 206; body was not read')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                     response.headers.get('Content-Range', ''), flags=re.IGNORECASE)
                if match is None:
                    raise FetchValidationError('Missing or malformed exact Content-Range')
                got_first, got_last, total = map(int, match.groups())
                if (got_first, got_last) != byte_range or total <= got_last:
                    raise FetchValidationError('Content-Range does not match requested bounds')
                if expected_total is not None and total != expected_total:
                    raise FetchValidationError('Safetensors total file length changed between ranges')
                record['total_file_bytes'] = total
            elif response.status_code != 200:
                raise FetchValidationError('Public metadata GET did not return HTTP 200')
            if response.headers.get('Content-Encoding', 'identity').lower() not in ('', 'identity'):
                raise FetchValidationError('Encoded response refused for exact byte accounting')
            content_length = response.headers.get('Content-Length')
            if content_length is not None:
                if not content_length.isdigit():
                    raise FetchValidationError('Invalid Content-Length')
                length = int(content_length)
                if length > limit or (expected_bytes is not None and length != expected_bytes):
                    raise FetchValidationError('Content-Length differs from the bounded request')
            body_limit = expected_bytes if expected_bytes is not None else limit
            digest = hashlib.sha256()
            with destination.open('xb') as stream:
                while True:
                    # Read at most one excess byte to detect an oversized body;
                    # never consume an unbounded full checkpoint response.
                    size = min(65536, body_limit - record['bytes_written'] + 1)
                    chunk = response.raw.read(size, decode_content=False)
                    if not chunk:
                        break
                    if record['bytes_written'] + len(chunk) > body_limit:
                        allowed = body_limit - record['bytes_written']
                        if allowed:
                            stream.write(chunk[:allowed]); digest.update(chunk[:allowed])
                            record['bytes_written'] += allowed
                        raise FetchValidationError('Response body exceeded its fixed byte bound')
                    stream.write(chunk)
                    digest.update(chunk)
                    record['bytes_written'] += len(chunk)
            if expected_bytes is not None and record['bytes_written'] != expected_bytes:
                raise FetchValidationError('Range body byte count differs from requested length')
            if content_length is not None and record['bytes_written'] != int(content_length):
                raise FetchValidationError('Response body is shorter than Content-Length')
            record.update(complete=True, sha256=digest.hexdigest())
            return record
        except BaseException as error:
            record['error_type'] = type(error).__name__
            if isinstance(error, FetchValidationError):
                record['error'] = str(error)
            if destination.exists():
                record['partial_sha256'] = sha256(destination)
                record['bytes_on_disk'] = destination.stat().st_size
            raise
        finally:
            if response is not None:
                response.close()
            record['finished_utc'] = utc_now()
            exclusive_json(self.requests_dir / f'{self.index:04d}.json', record)


def tensor_source_files(fetcher, output, revision, metadata):
    siblings = metadata.get('siblings')
    if not isinstance(siblings, list):
        raise FetchValidationError('Model metadata does not include repository filenames')
    filenames = {item.get('rfilename') for item in siblings if isinstance(item, dict)}
    if 'model.safetensors' in filenames:
        return {item['name']: 'model.safetensors' for item in TARGETS}
    if 'model.safetensors.index.json' not in filenames:
        raise FetchValidationError('No supported safetensors checkpoint or index is present')
    index_path = output / 'model.safetensors.index.json'
    fetcher.fetch(resolved_url(revision, index_path.name), index_path, limit=MAX_METADATA_BYTES)
    index = strict_json(index_path)
    mapping = index.get('weight_map', {})
    result = {}
    for target in TARGETS:
        filename = mapping.get(target['name'])
        if not isinstance(filename, str) or filename not in filenames or not filename.endswith('.safetensors'):
            raise FetchValidationError('Tensor index is missing an expected public safetensors shard')
        result[target['name']] = public_filename(filename)
    return result


def fetch_header(fetcher, output, revision, filename):
    # Shards may have directory prefixes; use a digest to avoid path collisions.
    label = hashlib.sha256(filename.encode()).hexdigest()[:12]
    directory = output / 'headers' / label
    directory.mkdir(parents=True)
    url = resolved_url(revision, filename)
    length_path = directory / 'length.bin'
    first = fetcher.fetch(url, length_path, limit=8, byte_range=(0, 7))
    total = first['total_file_bytes']
    header_size = struct.unpack('<Q', length_path.read_bytes())[0]
    if not 2 <= header_size <= MAX_HEADER_BYTES or 8 + header_size > total:
        raise FetchValidationError('Safetensors header length is invalid or exceeds the header limit')
    json_path = directory / 'header.json'
    fetcher.fetch(url, json_path, limit=header_size, byte_range=(8, 7 + header_size), expected_total=total)
    header_path = directory / 'header.bin'
    with header_path.open('xb') as stream:
        stream.write(length_path.read_bytes())
        stream.write(json_path.read_bytes())
    header = strict_json(json_path)
    if not isinstance(header, dict):
        raise FetchValidationError('Safetensors header must be a JSON object')
    record = {'source_file': filename, 'canonical_url': url, 'total_file_bytes': total,
              'data_start': 8 + header_size, 'header_json_bytes': header_size,
              'header_path': str(header_path.relative_to(output)), 'header_sha256': sha256(header_path),
              'header_json_path': str(json_path.relative_to(output)),
              'header_json_sha256': sha256(json_path)}
    exclusive_json(directory / 'manifest.json', record)
    return header, record


def fetch_weights(args, output):
    fetcher = PublicFetcher(output)
    try:
        metadata_path = output / 'repository.json'
        api_url = HF_BASE + '/api/models/' + MODEL_ID + '/revision/' + args.revision
        fetcher.fetch(api_url, metadata_path, limit=MAX_METADATA_BYTES)
        metadata = strict_json(metadata_path)
        revision = metadata.get('sha')
        if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise FetchValidationError('Model API did not return an exact 40-hex commit revision')
        if args.revision != 'main' and revision != args.revision:
            raise FetchValidationError('Resolved revision differs from the requested frozen commit')
        exclusive_json(output / 'resolved-revision.json', {'model_id': MODEL_ID, 'revision': revision,
                       'requested_revision': args.revision, 'api_url': api_url, 'resolved_utc': utc_now()})
        config_path, license_path = output / 'config.json', output / 'LICENSE'
        config_url = resolved_url(revision, 'config.json')
        license_url = resolved_url(revision, 'LICENSE')
        fetcher.fetch(config_url, config_path, limit=MAX_METADATA_BYTES)
        fetcher.fetch(license_url, license_path, limit=MAX_METADATA_BYTES)
        config = strict_json(config_path)
        if (config.get('model_type'), config.get('hidden_size'), config.get('intermediate_size')) != ('qwen3', 1024, 3072):
            raise FetchValidationError('Public config differs from the registered Qwen3-0.6B dimensions')
        if config.get('num_attention_heads', 0) * config.get('head_dim', 0) != 2048:
            raise FetchValidationError('Config attention projection width differs from 2048')
        if license_path.stat().st_size == 0:
            raise FetchValidationError('Public license is empty')
        files = tensor_source_files(fetcher, output, revision, metadata)
        headers = {}
        header_records = []
        for filename in sorted(set(files.values())):
            header, record = fetch_header(fetcher, output, revision, filename)
            headers[filename] = (header, record)
            header_records.append(record)
        (output / 'tensors').mkdir()
        tensors = []
        for target in TARGETS:
            filename = files[target['name']]
            header, source = headers[filename]
            tensor = header.get(target['name'])
            if not isinstance(tensor, dict):
                raise FetchValidationError('Expected tensor is absent from safetensors header')
            if tensor.get('dtype') != 'BF16' or tensor.get('shape') != target['shape']:
                raise FetchValidationError('Tensor dtype or shape differs from the frozen BF16 expectation')
            offsets = tensor.get('data_offsets')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(item) is not int for item in offsets)
                    or offsets[0] < 0 or offsets[1] <= offsets[0]):
                raise FetchValidationError('Tensor data offsets are malformed')
            byte_count = target['shape'][0] * target['shape'][1] * 2
            if offsets[1] - offsets[0] != byte_count:
                raise FetchValidationError('BF16 shape and tensor byte count do not agree')
            first = source['data_start'] + offsets[0]
            last = source['data_start'] + offsets[1] - 1
            if last >= source['total_file_bytes']:
                raise FetchValidationError('Tensor data range exceeds the safetensors file')
            tensor_path = output / target['path']
            response = fetcher.fetch(source['canonical_url'], tensor_path, limit=byte_count,
                                     byte_range=(first, last), expected_total=source['total_file_bytes'])
            record = {**target, 'dtype': 'BF16', 'byte_order': 'little',
                      'layout': 'C contiguous [output_features,input_features]; raw checkpoint weight',
                      'bytes': byte_count, 'sha256': response['sha256'], 'source_file': filename,
                      'data_offsets': offsets, 'file_byte_range': [first, last],
                      'canonical_url': source['canonical_url'],
                      'header_path': source['header_path'], 'header_sha256': source['header_sha256']}
            tensors.append(record)
            exclusive_json(output / 'tensors' / (tensor_path.stem + '.json'), record)
            print(json.dumps({'kind': 'tensor_fetched', 'name': target['name'],
                              'bytes': byte_count, 'sha256': record['sha256']}), flush=True)
        manifest = {'schema_version': 1, 'model_id': MODEL_ID, 'revision': revision,
                    'fetched_utc': utc_now(), 'repository_metadata_path': 'repository.json',
                    'repository_metadata_sha256': sha256(metadata_path),
                    'config_path': 'config.json', 'config_sha256': sha256(config_path),
                    'config_canonical_url': config_url,
                    'license_path': 'LICENSE', 'license_sha256': sha256(license_path),
                    'license_canonical_url': license_url,
                    'tensors': tensors, 'headers': header_records,
                    'validation': {'required_dtype': 'BF16', 'required_byte_order': 'little',
                                   'expected_shapes': {t['name']: t['shape'] for t in TARGETS},
                                   'http_range_status': 206, 'exact_content_range': True,
                                   'exact_bounded_byte_count': True, 'all_assertions_passed': True},
                    'scope': 'Actual public checkpoint weights only. No activations downloaded; any synthetic activations must be labeled separately. No model-quality result is implied.'}
        exclusive_json(output / 'manifest.json', manifest)
        return manifest
    finally:
        fetcher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--revision', default='main', help='main or an already frozen 40-hex commit')
    args = parser.parse_args()
    if args.revision != 'main' and not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        parser.error('--revision must be main or a lowercase 40-hex commit')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    status = {'started_utc': utc_now(), 'status': 'running', 'model_id': MODEL_ID,
              'requested_revision': args.revision, 'authentication': 'anonymous'}
    code = 1
    try:
        manifest = fetch_weights(args, output)
        status.update(status='completed', revision=manifest['revision'],
                      manifest_sha256=sha256(output / 'manifest.json'),
                      tensor_bytes=sum(t['bytes'] for t in manifest['tensors']))
        code = 0
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__)
        if isinstance(error, FetchValidationError):
            status['error'] = str(error)
        # Third-party network exception messages can contain signed CDN URLs.
        # Their class is sufficient for a safe first diagnosis; no traceback.
    finally:
        status['finished_utc'] = utc_now()
        exclusive_json(output / 'status.json', status)
        files = {str(path.relative_to(output)): {'bytes': path.stat().st_size,
                 'sha256': sha256(path)} for path in sorted(output.rglob('*')) if path.is_file()}
        exclusive_json(output / 'artifact-manifest.json', {'created_utc': utc_now(), 'files': files})
    print(json.dumps({'kind': 'weight_fetch_finished', 'output_dir': str(output), **status}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
