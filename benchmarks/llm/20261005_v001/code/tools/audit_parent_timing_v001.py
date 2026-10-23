"""Flag large synchronized-call timing excursions without removing any sample."""
import argparse,hashlib,json,statistics
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('inputs',type=Path,nargs='+');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    results=[]
    for source in a.inputs:
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        excursions=[];count=0
        for row in rows:
            if row.get('kind')!='layer' or 'samples_ms' not in row:continue
            for arm,times in row['samples_ms'].items():
                typical=statistics.median(times);count+=len(times)
                for i,t in enumerate(times):
                    if t>3*typical:excursions.append(dict(layer_zero_based=row['layer'],arm=arm,sample_index=i,ms=t,within_layer_median_ms=typical,ratio=t/typical))
        results.append(dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),samples=count,excursions=excursions))
    a.output.write_text(json.dumps(dict(rule='Flag >3x the same arm/layer sample median. Diagnostic only; all original samples and means are retained. No cause inferred.',results=results),indent=2)+'\n')
    print(json.dumps({r['source']:len(r['excursions']) for r in results}))


if __name__=='__main__':main()
