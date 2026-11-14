"""Compile the editable ACL manuscript; numerical results remain frozen."""
from pathlib import Path
import argparse, shutil, subprocess, tempfile

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--engine', default='tectonic', help='Path to Tectonic')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([args.engine, '--outdir', tmp, str(root/'report/latex/main.tex')],check=True)
        shutil.copy2(Path(tmp)/'main.pdf', root/'report/technical_report.pdf')
    print('Built report/technical_report.pdf')

if __name__=='__main__': main()
