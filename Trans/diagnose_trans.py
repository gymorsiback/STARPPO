import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Trans.train import TransformerActorCritic

def diagnose_trans_sensitivity():
    print("=== Trans Sensitivity Diagnosis ===")
    device = 'cpu'
    num_servers = 10
    embedding_dim = 64

    model = TransformerActorCritic(num_servers=num_servers, embedding_dim=embedding_dim).to(device)
    model.eval()

    base_feat = torch.zeros(1, num_servers, 7)

    feat_expensive = base_feat.clone()
    feat_expensive[:, :, 6] = 0.0 
    feat_cheap = base_feat.clone()
    feat_cheap[:, :, 6] = 5.0 
    with torch.no_grad():
        logits_exp, val_exp = model(feat_expensive)
        logits_cheap, val_cheap = model(feat_cheap)
    
    print("\nTest 1: Global Sensitivity")
    print(f"Logits (Expensive): {logits_exp[0, :3].numpy()}")
    print(f"Logits (Cheap)    : {logits_cheap[0, :3].numpy()}")
    print(f"Value (Expensive) : {val_exp.item():.4f}")
    print(f"Value (Cheap)     : {val_cheap.item():.4f}")
    
    diff = torch.abs(logits_exp - logits_cheap).mean().item()
    print(f"\n>> Mean Diff in Logits: {diff:.6f}")
    
    if diff < 1e-6:
        print("!! CRITICAL ALERT !! Model is completely blind to Cost feature!")
    else:
        print("Model can see Cost feature.")

    print("\nTest 2: Differential Sensitivity")
    feat_mixed = feat_expensive.clone()
    feat_mixed[:, 0, 6] = 5.0 
    
    with torch.no_grad():
        logits_mixed, _ = model(feat_mixed)

    probs = torch.softmax(logits_mixed, dim=-1)[0]
    print(f"Probabilities: {probs.numpy()}")
    print(f"Server 0 Prob: {probs[0]:.4f} (Should be higher if sensitive)")
    print(f"Others Prob  : {probs[1]:.4f}")
    
    if probs[0] > probs[1] + 0.01:
        print("Model prefers cheap server (Initial Random Weights)")
    else:
        print("Model shows no initial preference (Expected for untrained)")

if __name__ == "__main__":
    diagnose_trans_sensitivity()


