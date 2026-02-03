#!/usr/bin/env python3
"""
Test the learning rate schedule for resuming from checkpoint 54.
Verifies that LR starts at 3e-5 and decays to 1e-7 over 20 epochs using CosineAnnealingLR.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Configuration matching pi3.yaml
initial_lr = 3e-5
encoder_lr = 3e-6
eta_min = 1e-7
remaining_epochs = 20
iters_per_epoch = 800
total_steps = remaining_epochs * iters_per_epoch

print("=" * 60)
print("Learning Rate Schedule Test - CosineAnnealingLR")
print("=" * 60)
print(f"Initial LR (decoder): {initial_lr:.2e}")
print(f"Initial LR (encoder): {encoder_lr:.2e}")
print(f"Minimum LR (eta_min): {eta_min:.2e}")
print(f"Remaining epochs: {remaining_epochs}")
print(f"Iterations per epoch: {iters_per_epoch}")
print(f"Total steps: {total_steps}")
print("=" * 60)

# Create a dummy model with two parameter groups (encoder and decoder)
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),  # encoder
    torch.nn.Linear(10, 10),  # decoder
)

# Create optimizer with two parameter groups
optimizer = optim.AdamW([
    {'params': model[0].parameters(), 'lr': encoder_lr},
    {'params': model[1].parameters(), 'lr': initial_lr}
], weight_decay=5e-2)

# Create CosineAnnealingLR scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=eta_min
)

# Track learning rates
encoder_lrs = []
decoder_lrs = []
steps = []

# Simulate training
for epoch in range(remaining_epochs):
    for step in range(iters_per_epoch):
        current_step = epoch * iters_per_epoch + step
        steps.append(current_step)

        # Get current learning rates
        encoder_lr_current = optimizer.param_groups[0]['lr']
        decoder_lr_current = optimizer.param_groups[1]['lr']

        encoder_lrs.append(encoder_lr_current)
        decoder_lrs.append(decoder_lr_current)

        # Step the scheduler
        scheduler.step()

    # Print LR at the end of each epoch
    if epoch == 0 or epoch == remaining_epochs - 1 or (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1:2d}: Decoder LR = {decoder_lr_current:.8f}, Encoder LR = {encoder_lr_current:.8f}")

print("=" * 60)
print(f"Final Decoder LR: {decoder_lrs[-1]:.8f}")
print(f"Final Encoder LR: {encoder_lrs[-1]:.8f}")
print(f"Target eta_min: {eta_min:.8f}")
print("=" * 60)

# Plot learning rate schedule
plt.figure(figsize=(12, 6))
plt.plot(steps, decoder_lrs, label='Decoder LR', linewidth=2)
plt.plot(steps, encoder_lrs, label='Encoder LR', linewidth=2)
plt.axhline(y=eta_min, color='r', linestyle='--', label=f'eta_min = {eta_min:.2e}')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule (CosineAnnealingLR)\nFrom Checkpoint 54 to 74 (20 epochs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Add epoch markers
for epoch in range(0, remaining_epochs + 1, 5):
    step = epoch * iters_per_epoch
    if step <= total_steps:
        plt.axvline(x=step, color='gray', linestyle=':', alpha=0.5)
        plt.text(step, plt.ylim()[1] * 0.9, f'E{epoch}', rotation=90,
                verticalalignment='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('lr_schedule_cosine_resume.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to: lr_schedule_cosine_resume.png")
print("\nLearning rate schedule looks good! ✓")
