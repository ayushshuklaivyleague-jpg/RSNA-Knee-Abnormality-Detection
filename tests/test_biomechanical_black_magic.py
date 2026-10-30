import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import ndtri, ndtr

# Load Gold Ground Truth and OOF predictions
gt = pd.read_csv('data/gold_oof/gold_ground_truth.csv')
compact = pd.read_csv('data/gold_oof/gold_oof_compact.csv')
raptor = pd.read_csv('data/gold_oof/gold_oof_raptor.csv')

targets = [c for c in gt.columns if c != 'StudyInstanceUID']
uids = gt['StudyInstanceUID'].tolist()
n = len(uids)

# 1. Baseline 50/50 Uniform Rank Blend
aucs_baseline = []
for t in targets:
    rc = compact[t].rank(pct=True).values
    rr = raptor[t].rank(pct=True).values
    y = gt[t].values
    aucs_baseline.append(roc_auc_score(y, 0.5 * rr + 0.5 * rc))

print(f"=== BASELINE 50/50 Uniform Mean AUC: {np.mean(aucs_baseline):.5f} ===")

# 2. Latent Probit Mapping
def to_probit(df):
    ranks = df[targets].rank(method='average').values
    pct = (ranks - 0.5) / len(ranks)
    pct = np.clip(pct, 1e-5, 1.0 - 1e-5)
    return pd.DataFrame(ndtri(pct), columns=targets, index=df.index)

z_rap = to_probit(raptor)
z_comp = to_probit(compact)

# Conservative Surgical Routing Weights (W_RAPTOR)
W_RAPTOR = {
    'Lateral OA': 0.72,
    'Medial OA': 0.70,
    'Synovitis': 0.70,
    'Medial Meniscus': 0.72,
    'Fracture': 0.72,
    'Effusion': 0.65,
    'Lateral Meniscus': 0.50,
    'ACL': 0.28,
    'MCL': 0.28,
    'PF OA': 0.28,
    "Baker's": 0.28,
    'Contusion': 0.33,
}

# Fuse in Probit Space
z_fused = pd.DataFrame(index=gt.index)
for t in targets:
    wr = W_RAPTOR[t]
    z_fused[t] = wr * z_rap[t] + (1.0 - wr) * z_comp[t]

# 3. Apply Biomechanical Kinematic Cross-Priors
# A. Pivot-Shift ACL <-> Contusion
acl_lift = np.clip((z_fused['Contusion'] - 0.2) * 0.05, 0.0, 0.08)
z_fused['ACL'] = z_fused['ACL'] + acl_lift

# B. Degenerative Medial Meniscus <-> Medial OA
moa_lift = np.clip((z_fused['Medial Meniscus'] - 0.2) * 0.04, 0.0, 0.06)
z_fused['Medial OA'] = z_fused['Medial OA'] + moa_lift

# C. Hydrostatic Effusion <-> Synovitis & Baker's
syn_lift = np.clip((z_fused['Effusion'] - 0.2) * 0.04, 0.0, 0.06)
z_fused['Synovitis'] = z_fused['Synovitis'] + syn_lift
bak_lift = np.clip((z_fused['Effusion'] - 0.2) * 0.03, 0.0, 0.05)
z_fused["Baker's"] = z_fused["Baker's"] + bak_lift

# Convert back to probability via standard normal CDF
p_fused = pd.DataFrame(ndtr(z_fused.values), columns=targets, index=gt.index)

# Consensus Residual Tie-Breaking
raw_consensus = (raptor[targets] + compact[targets]) / 2.0
final_scores = p_fused + 1e-5 * raw_consensus

aucs_black_magic = []
for t in targets:
    y = gt[t].values
    s = final_scores[t].values
    auc_bm = roc_auc_score(y, s)
    aucs_black_magic.append(auc_bm)
    diff = auc_bm - aucs_baseline[targets.index(t)]
    print(f"{t:18s} | Baseline: {aucs_baseline[targets.index(t)]:.4f} | Black Magic: {auc_bm:.4f} | Gain: {diff:+.4f}")

print("-" * 65)
print(f"BASELINE MEAN AUC:     {np.mean(aucs_baseline):.5f}")
print(f"BLACK MAGIC MEAN AUC:  {np.mean(aucs_black_magic):.5f}")
print(f"NET LEADERBOARD BOOST: {np.mean(aucs_black_magic) - np.mean(aucs_baseline):+.5f} (+{(np.mean(aucs_black_magic) - np.mean(aucs_baseline))*100:.2f}%)")
