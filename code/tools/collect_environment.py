#!/usr/bin/env python3
"""Record hardware/software and immutable source hashes without training."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,json,platform,subprocess,sys

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result={'python':sys.version,'platform':platform.platform(),'packages':{}}
 for name in ['torch','torchvision','numpy','opencv-python','opencv-python-headless','h5py','diffusers','transformers']:
  try:result['packages'][name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
 import torch
 result['cuda_available']=torch.cuda.is_available();result['torch_cuda']=torch.version.cuda
 result['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
 try:result['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except subprocess.CalledProcessError:result['git_commit']=None
 root=Path(__file__).resolve().parents[1]
 result['python_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py') if '.git' not in f.parts}
 protocol_candidates=[
  root/'research_scripts'/'run_stage.sh',
  root/'scripts'/'run_stage.sh',
  root/'configs'/'controlled_protocol.json',
  *sorted(root.glob('requirements-*.txt')),
 ]
 result['protocol_sha256']={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in protocol_candidates if f.is_file()}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in {'python_sha256','protocol_sha256'}},indent=2))
if __name__=='__main__':main()
