"""Tiny CUDA diagnostic on the exact GPU stack, not Qwen/GRPO training."""
import copy,json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict
from src.training.prime_native import initialize_optimizer_state,optimizer_fingerprint
root=Path('/workspace/counter-diagnostic');root.mkdir(exist_ok=False)
torch.manual_seed(42)
model=torch.nn.Linear(2,1,device='cuda');reference=copy.deepcopy(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0)
ref_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-3,weight_decay=0)
initialize_optimizer_state(model,optimizer)
steps=[optimizer_fingerprint(optimizer,0)[1]]
assert steps[-1]==[0]
for version in range(3):
 weights,state=get_state_dict(model,[optimizer]);saved={'model':weights,'optimizer':state}
 dcp.save(saved,checkpoint_id=root/f'step-{version}')
 restored=torch.nn.Linear(2,1,device='cuda')
 opt=torch.optim.AdamW(restored.parameters(),lr=1e-3,weight_decay=0)
 weights,state=get_state_dict(restored,[opt]);loaded={'model':weights,'optimizer':state}
 dcp.load(loaded,checkpoint_id=root/f'step-{version}')
 set_state_dict(restored,[opt],model_state_dict=loaded['model'],optim_state_dict=loaded['optimizer'])
 assert optimizer_fingerprint(optimizer,version)==optimizer_fingerprint(opt,version)
 model,optimizer=restored,opt
 for actual,expected in zip(model.parameters(),reference.parameters(),strict=True):assert torch.equal(actual,expected)
 if version==2:break
 for learner,learner_opt in ((model,optimizer),(reference,ref_optimizer)):
  learner_opt.zero_grad(set_to_none=True)
  learner(torch.tensor([[1.,2.]],device='cuda')).sum().backward();learner_opt.step()
 steps.append(optimizer_fingerprint(optimizer,version+1)[1])
record={'diagnostic':'tiny_cuda_adamw_dcp_roundtrip','device':torch.cuda.get_device_name(),'torch':torch.__version__,'steps':steps,'roundtrips':3,'weights_match_fresh_adamw_at_every_step':True,'qwen_training':False,'grpo_training':False,'quality_improvement_measured':False}
(root/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
