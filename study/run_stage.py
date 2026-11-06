"""Bounded local subprocess pool; no duplicate job publication or overlapping outputs."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent

def execute(job):
    p=ROOT/'logs'/f'{job["id"]}.log';p.parent.mkdir(exist_ok=True)
    with p.open('a',encoding='utf-8') as out:
        # Resume completed candidates; archive incomplete checkpoints before restarting.
        for route in job['routes']:
            for folder in (ROOT/'results'/job['id']/route).glob('trial*'):
                if folder.is_dir() and not (folder/'fit.json').exists() and any(folder.iterdir()):
                    target=ROOT/'interrupted_candidates'/job['id']/route/(folder.name+'_'+str(time.time_ns()))
                    assert folder.resolve().is_relative_to(ROOT.resolve()) and target.resolve().is_relative_to(ROOT.resolve())
                    target.parent.mkdir(parents=True,exist_ok=True);folder.rename(target)
                    print('Archived incomplete candidate:',target.relative_to(ROOT),file=out,flush=True)
        done=subprocess.run([sys.executable,'-u',str(ROOT/'worker.py'),'--job',job['id']],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    return job['id'],done.returncode

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=list('ADBCE'));p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    assert 1<=args.workers<=3
    jobs=[j for j in json.loads((ROOT/'jobs.json').read_text()) if j['stage']==args.stage]
    start=time.perf_counter();failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(execute,j) for j in jobs]):
            name,code=future.result();print(name,'exit',code,flush=True)
            if code:failures.append(name)
    result={'stage':args.stage,'jobs':len(jobs),'failures':failures,'wall_seconds':time.perf_counter()-start}
    (ROOT/f'stage_{args.stage}_execution.json').write_text(json.dumps(result,indent=2))
    if failures:raise SystemExit('Failed jobs retained: '+','.join(failures))
    print('Stage complete:',json.dumps(result),flush=True)

if __name__=='__main__':main()
