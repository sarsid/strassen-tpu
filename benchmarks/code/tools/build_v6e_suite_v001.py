"""Freeze a shape-aware v6e search before any 168-shape measurements."""
import argparse,copy,hashlib,json
from pathlib import Path

def build(root,out):
    out.mkdir(parents=True,exist_ok=False)
    source=root/'configs/mlsys_shapes_v001/shapes.json'
    manifest=json.loads(source.read_text())
    assert len(manifest['shapes'])==168
    cfg=copy.deepcopy(json.loads((root/'configs/v6e_ten_v001/campaign.json').read_text()))
    tiles=[[32,512,512],[32,1024,2048],[32,4096,2048],[128,512,512],
           [128,1024,2048],[256,1024,512],[512,1024,1024],[1024,512,1024],
           [768,1024,512],[1536,1536,512],[1024,1024,1024],[2048,1024,1024],
           [2048,2048,2048],[1024,4096,2048],[4096,1024,2048],
           [1024,2048,4096],[2048,1024,4096],[512,2048,2048],
           [1024,1024,4096],[4096,2048,1024]]
    families={'native':cfg['candidate_families']['native'],'cubic':[], 'one_level':[], 'two_level':[]}
    for tile in tiles:
        for family,impl,depth,mode in [('cubic','cubic',0,None),
                ('one_level','pipeline',1,'panel'),('one_level','pipeline',1,'deferred'),
                ('one_level','original',1,None),('two_level','pipeline',2,'hybrid'),
                ('two_level','original',2,None)]:
            cid='_'.join([family,impl,mode or 'base',*map(str,tile)])
            c=dict(candidate_id=cid,algorithm='cubic_full' if depth==0 else 'strassen',
                   variant='plain',tile=tile,implementation=impl,depth=depth,
                   compiler_options={},vmem_limit_bytes=112*1024**2,order='interleaved')
            if impl=='pipeline':c.update(mode=mode,buffers=2,traversal='mn')
            families[family].append(c)
    cfg.update(campaign_id='v6e_suite_v001',arms=list(families),candidate_families=families,
        tile_pool=tiles,order_seed=239168,screen_seed=2026092400,
        confirm_seeds=[2026102400,2026112400,2026122400],llo_instrumentation=False,
        candidate_policy='Rank common depth-valid tiles by padded/useful volume, then descending tile volume, then tile tuple. Four tiles if any dimension <1024, six otherwise. All revised modes and matched cubic on these tiles; original implementations on the first two. This rule uses dimensions only, never timings.',
        selection_policy='Fastest eligible screening mean per family; default Native and exact tile-matched cubic controls for both frozen depths added to confirmation. Never reselect on confirmation.',
        development_set=True,unseen_shape_validation=False,
        qualification_shapes=[dict(id='qual_tiny',m=1,k=1,n=1),
            dict(id='qual_skinny',m=16,k=4096,n=4096),
            dict(id='qual_boundary',m=1025,k=1025,n=1025),
            dict(id='qual_multi_panel',m=2049,k=4097,n=1025),
            dict(id='qual_large',m=8192,k=8192,n=8192)])
    cfg['timing']['screen']={'warmups':3,'repeats':8}
    cfg['timing']['smoke']={'warmups':0,'repeats':1}
    cfg['memory']['execution_order']='12 archived batches of 14 shapes; all arms paired within a shape; compile reuse across three fresh confirmation inputs; device profiles separate.'
    cfg.pop('profile_shapes',None)
    manifest['v6e_provenance']=dict(source=str(source.relative_to(root)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        policy='All 168 original shapes retained in original order. No performance-based shape filtering.')
    for name,data in [('campaign.json',cfg),('shapes.json',manifest),('distributions.json',
            json.loads((root/'configs/v6e_ten_v001/distributions.json').read_text()))]:
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(shapes=168,pool_tiles=len(tiles),registered_candidates=sum(map(len,families.values())),output=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();build(a.root,a.output_dir)
