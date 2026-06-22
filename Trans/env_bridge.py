
import os
import sys

# Add project root to path to import env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from env import WorkflowDataset, WorkflowMoEEnv
from config.Params import configs

def get_env(batch_size=1):
    """
    Creates a WorkflowMoEEnv instance adapted for Trans algorithm.
    Wrapper to match ET's expectation if needed, or just return standard env.
    """
    dataset = WorkflowDataset(data_root=os.path.join(PROJECT_ROOT, 'data', 'fuzz_test_data'), split='train')
    env = WorkflowMoEEnv(dataset, device='cuda') # Using cuda by default for training
    return env


