#!/usr/bin/env python3
"""
Diagnose what causes depth_loss = 51.65

Given:
- depth_loss = 51.647213
- Average weights ~ 8.88 (from simulation)
- Depth range: [0.2679, 1.5114], mean: 0.7129

Reverse engineer the actual depth error.
"""

import torch

print("=" * 80)
print("Reverse Engineering Depth Error from Loss Value")
print("=" * 80)

# Given values
depth_loss_observed = 51.647213
typical_weight_mean = 8.88  # from simulation

# Reverse calculation
# depth_loss = mean(weights * |pred - gt|)
# If weights ~ 8.88, then |pred - gt| ~ depth_loss / weights
unweighted_error_estimate = depth_loss_observed / typical_weight_mean

print(f"\nGiven:")
print(f"  Observed depth_loss: {depth_loss_observed:.4f}")
print(f"  Typical weight mean (from simulation): {typical_weight_mean:.2f}")

print(f"\nEstimated unweighted depth error:")
print(f"  |pred - gt| ≈ {unweighted_error_estimate:.4f} meters")

print(f"\nContext:")
print(f"  GT depth range: [0.2679, 1.5114], mean: 0.7129")
print(f"  Error ratio: {unweighted_error_estimate / 0.7129 * 100:.1f}% of mean depth")

print("\n" + "=" * 80)
print("POSSIBLE CAUSES OF LARGE ERROR:")
print("=" * 80)

causes = [
    ("1. Scale mismatch",
     "Predictions and GT are in different scales due to normalization issues"),

    ("2. Network not learning yet",
     "Early training iterations - network outputs are still random/uninitialized"),

    ("3. Depth activation overflow",
     "If using exp(z), large z values cause depth to explode. You're using softplus, so less likely."),

    ("4. Intrinsics scaling issue",
     "If GT is normalized but intrinsics aren't scaled correctly, variance computation is wrong"),

    ("5. Very small variance regions",
     "If actual data has smaller variance than simulation, weights could be 10-100x larger"),
]

for title, desc in causes:
    print(f"\n{title}:")
    print(f"  {desc}")

print("\n" + "=" * 80)
print("DIAGNOSTIC STEPS:")
print("=" * 80)

steps = [
    "1. Check if predictions are in correct range (should be ~[0.3, 1.5])",
    "2. Look at the [DEBUG] outputs that should print occasionally (1% chance)",
    "3. Check if norm_factor is being applied correctly to both GT and intrinsics",
    "4. Visualize pred vs gt depth side by side",
    "5. Check actual variance values (sigma_Z2) on real training data",
]

for step in steps:
    print(f"  {step}")

print("\n" + "=" * 80)
print("QUICK FIXES TO TRY:")
print("=" * 80)

fixes = [
    ("1. Increase loss epsilon",
     "In loss_ablation.py line 289-290, change:\n"
     "     sigma_std = torch.sqrt(sigma_Z2 + 1e-3)  # increase from 1e-6\n"
     "     weights = 1.0 / (sigma_std + 1e-3)  # increase from 1e-6"),

    ("2. Clamp maximum weight",
     "Add after line 290:\n"
     "     weights = torch.clamp(weights, max=50.0)"),

    ("3. Scale down depth loss",
     "Add a scaling factor:\n"
     "     depth_loss = weighted_depth_loss.mean() * 0.1  # scale down by 10x"),

    ("4. Use simpler weighting",
     "Replace variance weighting with simpler depth weighting:\n"
     "     weights = 1.0 / (gt_depth + 0.1)  # similar to original Pi3Loss"),
]

for i, (title, desc) in enumerate(fixes, 1):
    print(f"\n{title}:")
    for line in desc.split('\n'):
        print(f"  {line}")

print("\n" + "=" * 80)
print("RECOMMENDED ACTION:")
print("=" * 80)
print("""
The most likely cause is #1 (scale mismatch) or #5 (smaller variance in real data).

IMMEDIATE FIX:
1. Increase epsilon to stabilize weights:
   - Change sigma_std = sqrt(sigma_Z2 + 1e-3)  # was 1e-6
   - Change weights = 1.0 / (sigma_std + 0.01)  # was 1e-6

2. Or add weight clamp:
   - weights = torch.clamp(weights, max=50.0)

This should bring depth_loss down from ~50 to ~5-10 range.
""")
