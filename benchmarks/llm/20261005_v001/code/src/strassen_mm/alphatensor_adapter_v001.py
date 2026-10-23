"""Load exact released AlphaTensor functions; explicitly label one dtype change.

Upstream files and Apache-2.0 notice are preserved in third_party/. Only three
function definitions are loaded from utils.py so its unrelated dm-tree timing
helpers and GPU launcher need not be imported or run. The released arithmetic
AST is unchanged in upstream mode. The second mode changes exactly one @ node
to a DEFAULT dot with FP32 preferred output; recombination then remains FP32.
"""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Callable, List

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'third_party/alphatensor_1949163_v001'


def load(fp32_recombine=False):
    receipt = json.loads((VENDOR / 'UPSTREAM.json').read_text())
    for name, expected in receipt['files'].items():
        assert hashlib.sha256((VENDOR/name).read_bytes()).hexdigest() == expected, name
    spec = importlib.util.spec_from_file_location('alphatensor_factors', VENDOR/'benchmarking/factorizations.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factors = module.get_4x4x4_alphatensor_tpu()
    tree = ast.parse((VENDOR/'benchmarking/utils.py').read_text())
    names = {'block_split', 'get_matrix_multiplication_tensor', 'algorithm_from_factors'}
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(tree.body) == 3
    if fp32_recombine:
        class ChangeDot(ast.NodeTransformer):
            count = 0
            def visit_BinOp(self, node):
                if isinstance(node.op, ast.MatMult):
                    self.count += 1
                    replacement = ast.parse('jnp.dot(left, right, precision="DEFAULT", preferred_element_type=jnp.float32)', mode='eval').body
                    return ast.copy_location(replacement, node)
                return self.generic_visit(node)
        changer = ChangeDot()
        tree = changer.visit(copy.deepcopy(tree))
        assert changer.count == 1
    namespace = dict(jnp=jnp, np=np, Callable=Callable, List=List, BlockMatrix=List[List[jnp.ndarray]])
    exec(compile(ast.fix_missing_locations(tree), str(VENDOR/'benchmarking/utils.py'), 'exec'), namespace)
    expected = namespace['get_matrix_multiplication_tensor'](4)
    np.testing.assert_array_equal(np.einsum('ir,jr,kr->ijk', *factors), expected)
    return factors, namespace


def make_matmul(shape, *, fp32_recombine=False):
    if any(d % 4 for d in shape):
        raise ValueError('This proof of concept requires all dimensions divisible by four')
    factors, functions = load(fp32_recombine)
    algorithm = functions['algorithm_from_factors'](factors)
    split = functions['block_split']
    def complete(a, b):
        blocks = algorithm(split(a, 4, 4), split(b, 4, 4))
        return jnp.concatenate([jnp.concatenate(row, axis=1) for row in blocks], axis=0)
    complete.metadata = {
        'upstream_commit': '1949163da3bef7e3eb268a3ac015fd1c2dbfc767',
        'factorization': 'get_4x4x4_alphatensor_tpu', 'rank': 49,
        'shape_mkn': list(shape), 'padded_shape_mkn': list(shape),
        'input_dtype': 'bfloat16', 'pre_add_dtype': 'bfloat16',
        'dot_output_and_recombination_dtype': 'float32' if fp32_recombine else 'bfloat16',
        'output_dtype': 'float32' if fp32_recombine else 'bfloat16',
        'adaptation': 'dot preferred_element_type=float32' if fp32_recombine else 'released arithmetic unchanged',
        'complete_call_scope': 'full device arrays -> split -> released algebra -> assemble full device output',
        'tiling': 'four by four blocks of entire operands; XLA manages each subproduct',
    }
    return complete
