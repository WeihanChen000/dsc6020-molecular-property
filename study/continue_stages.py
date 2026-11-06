"""Run remaining stages with verification gates and a resource-only optional-stage decision."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def command(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[1,2,3],default=3);args=p.parse_args()
    # D is launched separately so its finite budget can be inspected while running.
    command('verify.py','--stage','D')
    for stage in ['B','C']:
        command('run_stage.py',stage,'--workers',str(args.workers));command('verify.py','--stage',stage)
    import psutil
    times=[]
    for p in (ROOT/'results').glob('B_*/ridge/trial*/fit.json'):times.append(json.loads(p.read_text())['fit_seconds'])
    # Conservative width allowance for 978-d fusion vs 768-d LM; do not inspect scores.
    estimate=max(times)*196*2 if times else float('inf')
    available=psutil.virtual_memory().available/1024**3
    decision={'available_memory_GiB':available,'conservative_optional_fit_seconds':estimate,'rule':'All required verifications pass, available memory > 2 GiB, estimated added summed CPU fit time < 600 s','execute':bool(available>2 and estimate<600)}
    (ROOT/'optional_stage_decision.json').write_text(json.dumps(decision,indent=2))
    if decision['execute']:command('run_stage.py','E','--workers',str(args.workers));command('verify.py','--stage','E')
    command('verify.py','--stage','all')
    command('analyze.py')
    print('Remaining stages and analysis complete',flush=True)
if __name__=='__main__':main()
