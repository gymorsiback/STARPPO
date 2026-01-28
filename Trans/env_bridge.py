
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env import WorkflowDataset, WorkflowMoEEnv
from config.Params import configs

def get_env(batch_size=1):
    dataset = WorkflowDataset(data_root=os.path.join(PROJECT_ROOT, 'data', 'fuzz_test_data'), split='train')
    env = WorkflowMoEEnv(dataset, device='cuda') 
    return env


