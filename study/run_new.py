"""Re-run fixed v3 jobs in a fresh timestamped directory; original evidence is preserved."""
from pathlib import Path
import argparse,datetime,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,choices=[1,2,3],default=3);args=p.parse_args()
    dest=ROOT/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');dest.mkdir(parents=True,exist_ok=False)
    for name in ['core.py','worker.py','run_stage.py','verify.py','continue_stages.py','analyze.py','structure_diagnostics.py','audit_provenance.py','protocol.json','jobs.json','input_manifest.json','inherited_manifest.json','training_membership.csv','split_statistics.csv','实验协议.md','requirements-lock.txt']:
        shutil.copy2(ROOT/name,dest/name)
    for name in ['inputs','inherited_v2','provenance']:shutil.copytree(ROOT/name,dest/name)
    def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=dest,check=True)
    print('Fresh run:',dest,flush=True);run('audit_provenance.py')
    run('run_stage.py','A','--workers',str(args.workers));run('verify.py','--stage','A');run('run_stage.py','D','--workers',str(args.workers));run('continue_stages.py','--workers',str(args.workers));run('structure_diagnostics.py')
    print('Completed fresh downstream reproduction:',dest,flush=True)
if __name__=='__main__':main()
