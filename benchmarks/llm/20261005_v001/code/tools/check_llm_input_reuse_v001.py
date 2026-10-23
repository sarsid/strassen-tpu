"""Exercise first-build/reuse of model inputs without network or pretrained weights."""
import argparse,hashlib,json,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
from llm_queue_state_v002 import read,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True)
    import numpy as np,transformers,huggingface_hub
    import prepare_llm_case_v002 as original
    import prepare_llm_case_v003 as cached
    root=Path(__file__).resolve().parents[1];protocol=root/'configs/llm_v6e_v002.json';model=read(protocol)['models'][0];checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name);print(json.dumps(dict(check=name,status='passed')),flush=True)
    with tempfile.TemporaryDirectory() as temp:
        t=Path(temp);private=t/'private';private.mkdir();tokens=np.arange(20000,dtype=np.int32)%97
        first=t/'first';second=t/'second';calls=[]
        def legacy():
            calls.append('oracle');first.mkdir();(first/'official').mkdir();np.save(first/'official/layer-000-output.npy',np.zeros((1,64,4),np.float32));np.save(first/'oracle-tokens.npy',tokens[:64].reshape(1,64))
            save(first/'checkpoint-provenance.json',{'fixture':True});save(first/'summary.json',dict(model=model,oracle={'fixture':True},batch=1,sequence=512,artifacts={}))
            for split in ('train','test'):
                np.save(first/(split+'-tokens.npy'),tokens[:512].reshape(1,512));save(first/(split+'-tokens.json'),dict(split=split,shape=[1,512],token_sha256='initial',text_sha256=hashlib.sha256(b'fixture').hexdigest()))
        dataset=types.ModuleType('datasets');dataset.load_dataset=lambda *args,**kwargs:[dict(text='fixture')]
        tokenizer=lambda text,**kwargs:dict(input_ids=tokens.tolist())
        def argv(output,batch,sequence,model_id=model['id']):return ['prepare','--protocol',str(protocol),'--model',model_id,'--private',str(private),'--output',str(output),'--batch',str(batch),'--sequence',str(sequence)]
        with patch.dict(sys.modules,datasets=dataset),patch.object(original,'main',side_effect=legacy),patch.object(huggingface_hub,'snapshot_download',return_value=str(t)),patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=tokenizer),patch.object(sys,'argv',argv(first,1,512)):cached.main()
        check('first workload builds one official oracle and full corpus cache',len(calls)==1 and (private/'input-template/cache-receipt.json').exists())
        with patch.object(original,'main',side_effect=AssertionError('must not rebuild oracle')),patch.object(sys,'argv',argv(second,4,1024)):cached.main()
        check('second workload uses exact larger slice without repetition',np.array_equal(np.load(second/'train-tokens.npy'),tokens[:4096].reshape(4,1024)))
        check('oracle and checkpoint provenance preserved',cached.digest(first/'official/layer-000-output.npy')==cached.digest(second/'official/layer-000-output.npy') and read(second/'summary.json')['reused_model_inputs'])
        check('new workload token hash and shape recorded',read(second/'test-tokens.json')['shape']==[4,1024] and read(second/'test-tokens.json')['token_sha256']==hashlib.sha256(tokens[:4096].tobytes()).hexdigest())
        with patch.object(sys,'argv',argv(t/'wrong-model',1,512,read(protocol)['models'][1]['id'])):
            try:cached.main()
            except ValueError:check('different model cannot reuse cached inputs',True)
            else:check('different model rejected',False)
        (private/'input-template/official/layer-000-output.npy').write_bytes(b'corrupt')
        with patch.object(sys,'argv',argv(t/'corrupt',1,512)):
            try:cached.main()
            except ValueError:check('corrupt oracle cache rejected before output',not (t/'corrupt').exists())
            else:check('corrupt cache rejected',False)
    summary=dict(status='passed',checks=checks,scope='actual preparation wrapper, external download/tokenization/oracle construction substituted')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
