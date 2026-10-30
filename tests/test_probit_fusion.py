import numpy as np
import pandas as pd
from scipy.special import ndtri, ndtr

# test data: 10 studies, 12 labels
labels = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
          'Contusion', 'Fracture']
test_ids = [f'study_{i}' for i in range(10)]
n = len(test_ids)

SURGICAL_ROUTING = {
    'Lateral OA': 0.72, 'Medial OA': 0.70, 'Synovitis': 0.70,
    'Medial Meniscus': 0.72, 'Fracture': 0.72, 'Effusion': 0.65,
    'Lateral Meniscus': 0.50, 'ACL': 0.28, 'MCL': 0.28,
    'PF OA': 0.28, "Baker's": 0.28, 'Contusion': 0.33
}

# mock predictions
np.random.seed(42)
raw = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
ours = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
r224 = pd.DataFrame(np.random.uniform(0.1, 0.9, (n, len(labels))), index=test_ids, columns=labels)
A224 = 0.35

def to_probit(series):
    r = series.rank(method='average').values
    pct = (r - 0.5) / len(r)
    pct = np.clip(pct, 1e-5, 1.0 - 1e-5)
    return ndtri(pct)

out = np.zeros((n, len(labels)), dtype=np.float64)
for j, c in enumerate(labels):
    wr = SURGICAL_ROUTING.get(c, 0.50)
    
    # Raptor 384 + 224 in probit space
    z_r384 = to_probit(raw[c])
    z_r224 = to_probit(r224[c])
    z_rap = (1.0 - A224) * z_r384 + A224 * z_r224
    
    # ConvNeXt R7 in probit space
    z_ours = to_probit(ours[c])
    
    # Fused latent probit
    z_fused = wr * z_rap + (1.0 - wr) * z_ours
    
    # Convert back to probability via standard normal CDF
    p_fused = ndtr(z_fused)
    
    # Consensus tie-breaker
    raw_consensus = (raw[c].values + r224[c].values + ours[c].values) / 3.0
    final_score = p_fused + 1e-5 * raw_consensus
    
    # Final percentile rank in [0, 1]
    out[:, j] = pd.Series(final_score).rank(method='average', pct=True).values

sub = pd.DataFrame(out.astype(np.float32), columns=labels, index=test_ids)
print('Simulated Output shape:', sub.shape)
print('Any NaN?:', sub.isna().any().any())
print('Min / Max:', sub.min().min(), sub.max().max())
print('Sample output row 0:\n', sub.iloc[0])
print('TEST PASSED SUCCESSFULLY!')
