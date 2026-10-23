"""Ten historical winners after completed v6e architecture diagnostics."""
import prepare_v6e_opt_v001 as previous

def build(args):
    plan=previous.build(args);plan['title']='Ten historical v5e wins on v6e after diagnostics'
    for stage in plan['stages']:
        if stage['operation']=='phase':
            stage['campaign']='configs/v6e_ten_v001/campaign.json'
            stage['module']='strassen_mm.benchmark_v6e_ten_v001'
            stage.pop('expected',None)
            if stage['id']=='opt-screen':stage['requires']=['opt-smoke'];stage['timeout_seconds']=14400
            if stage['id']=='opt-confirm':stage['timeout_seconds']=10800
        if stage['id']=='report':stage['command'][1]='{source}/tools/report_v6e_ten_v001.py'
    plan['stages']=[s for s in plan['stages'] if s['id']!='opt-profile']
    plan['decisions']=[
      'Ten shapes frozen from historical v5e results BEFORE any new v6e timing. Both depths had clear paired pointwise 95% wins versus both Native baselines.',
      'Separate completed architecture diagnostic informs candidate set. Current JAX 0.11.2/jaxlib 0.11.2/libtpu 0.0.48.',
      'Only depths1/2. Original implementations, architecture-driven candidates, classical control and five Native settings.',
      'All screen arms use the same inputs and randomized paired timing rounds per shape, avoiding separate temporal batches.',
      'Select numerically passing fastest screen candidate per family, freeze, then confirm on three fresh Gaussian inputs with thirty paired rounds each. No reselection.',
      'Same BF16 operands/preadds, FP32 accumulation/output and fixed numerical gates. Sampled all-K FP64 error and full-output finiteness.',
      'Synchronized complete-call timings include device padding and cropping. Transfers/compilation excluded. Device traces collected separately, without LLO instrumentation.',
      'Synthetic inputs on documented winning dimensions; no claim of a new real-weight/full-model inference benchmark.',
      'One owned allocation, serial execution, immutable artifacts, retrieve and automatically release.']
    return plan
previous.campaign.build_plan=build
if __name__=='__main__':previous.campaign.main()
