"""Algebra, masking, numerical-contract and ablation tests, using CPU interpretation."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from strassen_mm import kernels_v002 as old
from strassen_mm import strassen_optimized as opt


class OptimizedContracts(unittest.TestCase):
    def test_invalid_configs_fail_early(self):
        for kwargs in ({'variant':'other'},{'traversal':'mkn'},{'product_order':[1]*7},
                       {'product_order':[True,2,3,4,5,6,7]},{'product_order':None},{'vmem_limit_bytes':0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):opt.describe((17,257,259),(16,256,256),**kwargs)
        with self.assertRaises(ValueError):opt.describe((0,256,256),(16,256,256))
        with self.assertRaises(ValueError):opt.describe((16,256,256),(16,128,256))

    def test_dtype_and_axis_contracts(self):
        fn=opt.make_matmul((17,257,259),(16,256,256),interpret=True)
        self.assertEqual(fn.metadata['shape_mkn'],[17,259,257])
        self.assertEqual(fn.metadata['padded_shape_mkn'],[17,259,257])
        self.assertEqual(fn.metadata['logical_padded_shape_mkn'],[32,512,512])
        with self.assertRaisesRegex(ValueError,'BF16'):fn(jnp.ones((17,259)),jnp.ones((259,257)))
        with self.assertRaises(ValueError):fn.finish(jnp.ones((17,259)))

    def test_masks_remove_external_padding_not_tile_arithmetic(self):
        baseline=opt.describe((1025,1025,1025),(1024,1024,512),variant='baseline')
        masked=opt.describe((1025,1025,1025),(1024,1024,512),variant='masked_edges')
        self.assertEqual(baseline['source_dot_flops'],masked['source_dot_flops'])
        self.assertGreater(masked['avoided_global_padding_extent_bytes'],0)
        self.assertFalse(masked['padding_required']);self.assertTrue(masked['out_of_bounds_masking_required'])
        self.assertEqual(masked['argument_and_output_bytes'],8*1025**2)

    def test_local_accumulator_counts_and_first_panel(self):
        for k in (256,768):
            a=opt.describe((16,256,k),(16,256,256),variant='baseline')
            b=opt.describe((16,256,k),(16,256,256),variant='local_accumulators')
            self.assertEqual(a['source_accumulator_element_additions']-b['source_accumulator_element_additions'],16*256)
            self.assertEqual(b['source_output_quadrant_writes_per_panel'],4)

    def test_peeled_work_is_disjoint_core_plus_tails(self):
        m,n,k=17,257,259;r=opt.describe((m,n,k),(16,256,256),variant='peeled_edges')
        core=16*256*256
        self.assertEqual(r['core_shape_mnk'],[16,256,256])
        self.assertEqual(r['source_dot_flops'],7*core//4+2*(m*n*k-core))
        self.assertFalse(r['native_only_fallback']);self.assertEqual(r['algorithm'],'hybrid_strassen_native')
        for s in ((1,257,259),(17,1,259),(17,257,1)):
            r=opt.describe(s,(16,256,256),variant='peeled_edges')
            self.assertTrue(r['native_only_fallback']);self.assertEqual(r['source_dot_flops'],2*np.prod(s))


class OptimizedAlgebra(unittest.TestCase):
    @staticmethod
    def inputs(shape,seed=933,random=False):
        m,n,k=shape;rng=np.random.default_rng(seed)
        if random:a,b=rng.normal(size=(m,k)),rng.normal(size=(k,n))
        else:a,b=rng.integers(-2,3,size=(m,k)),rng.integers(-2,3,size=(k,n))
        return jnp.asarray(a,jnp.bfloat16),jnp.asarray(b,jnp.bfloat16)

    def test_exact_algebra_all_variants_traversals_and_edge_axes(self):
        # These include a partial block in every axis, multiple K panels,
        # a pure-native peeled fallback, and garbage-padded interpreter refs.
        for shape in ((16,256,256),(17,257,259),(3,129,5),(31,513,257),
                      (7,127,127),(9,129,129),(15,255,255)):
            a,b=self.inputs(shape)
            ref=np.asarray(a,dtype=np.float32)@np.asarray(b,dtype=np.float32)
            for variant in opt.VARIANTS:
                for traversal in opt.TRAVERSALS:
                    with self.subTest(shape=shape,variant=variant,traversal=traversal):
                        fn=opt.make_matmul(shape,(16,256,256),variant=variant,traversal=traversal,interpret=True)
                        got=jax.jit(fn)(a,b)
                        self.assertEqual(got.dtype,jnp.float32)
                        np.testing.assert_array_equal(np.asarray(got),ref)

    def test_product_orders_preserve_integer_algebra(self):
        shape=(17,257,513);a,b=self.inputs(shape)
        ref=np.asarray(a,dtype=np.float32)@np.asarray(b,dtype=np.float32)
        for order in (tuple(range(1,8)),tuple(range(7,0,-1)),(6,4,7,3,1,2,5)):
            for variant in ('optimized','peeled_edges'):
                with self.subTest(order=order,variant=variant):
                    fn=opt.make_matmul(shape,(16,256,256),product_order=order,variant=variant,interpret=True)
                    np.testing.assert_array_equal(np.asarray(jax.jit(fn)(a,b)),ref)

    def test_new_padded_control_matches_frozen_kernel(self):
        shape=(17,257,513);m,n,k=shape;a,b=self.inputs(shape,random=True)
        before=old.make_matmul('strassen',(m,k,n),(16,256,256),variant='interleaved_output_accumulator',interpret=True)
        after=opt.make_matmul(shape,(16,256,256),variant='baseline',interpret=True)
        np.testing.assert_array_equal(np.asarray(jax.jit(before)(a,b)),np.asarray(jax.jit(after)(a,b)))

    def test_random_bf16_gate_and_prepare_kernel_finish_consistency(self):
        shape=(17,257,259);a,b=self.inputs(shape,random=True)
        ref=np.asarray(a,dtype=np.float64)@np.asarray(b,dtype=np.float64)
        for variant in opt.VARIANTS:
            with self.subTest(variant=variant):
                fn=opt.make_matmul(shape,(16,256,256),variant=variant,interpret=True)
                got=jax.jit(fn)(a,b);out=np.asarray(got,dtype=np.float64)
                self.assertTrue(np.isfinite(out).all())
                self.assertLess(np.linalg.norm(out-ref)/np.linalg.norm(ref),.02)
                self.assertLessEqual(np.max(np.abs(out-ref)),.001+.05*np.max(np.abs(ref)))
                prepared=fn.prepare(a,b)
                self.assertEqual(list(prepared[0].shape),[fn.metadata['padded_shape_mkn'][0],fn.metadata['padded_shape_mkn'][1]])
                reconstructed=fn.finish(jax.jit(fn.kernel)(*prepared))
                np.testing.assert_array_equal(np.asarray(got),np.asarray(reconstructed))

    def test_first_panel_never_reads_uninitialized_accumulators(self):
        # A zero product must overwrite even NaN-initialized interpreter outputs.
        for shape in ((16,256,256),(17,257,259)):
            m,n,k=shape;a=jnp.zeros((m,k),jnp.bfloat16);b=jnp.ones((k,n),jnp.bfloat16)
            for variant in ('local_accumulators','optimized','peeled_edges'):
                fn=opt.make_matmul(shape,(16,256,256),variant=variant,interpret=True)
                np.testing.assert_array_equal(np.asarray(jax.jit(fn)(a,b)),np.zeros((m,n),np.float32))


if __name__=='__main__':unittest.main(verbosity=2)
