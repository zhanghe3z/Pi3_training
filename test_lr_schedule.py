#!/usr/bin/env python3
"""
Test script to verify the learning rate schedule after resuming from checkpoint_54
"""
import torch
from torch.optim.lr_scheduler import OneCycleLR

# Configuration
max_lr = 5e-5
div_factor = 1000.0
final_div_factor = 100.0
pct_start = 0.0
num_epochs = 74
iters_per_epoch = 800
resume_epoch = 54

total_steps = num_epochs * iters_per_epoch
resume_step = resume_epoch * iters_per_epoch
remaining_steps = (num_epochs - resume_epoch) * iters_per_epoch

print("=" * 60)
print("Learning Rate Schedule Test")
print("=" * 60)
print(f"Total epochs: {num_epochs}")
print(f"Iterations per epoch: {iters_per_epoch}")
print(f"Total steps: {total_steps}")
print(f"Resume from epoch: {resume_epoch}")
print(f"Resume step: {resume_step}")
print(f"Remaining epochs: {num_epochs - resume_epoch}")
print(f"Remaining steps: {remaining_steps}")
print("=" * 60)

# Create a dummy model and optimizer
model = torch.nn.Linear(10, 10)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# Create scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    pct_start=pct_start,
    anneal_strategy='cos',
    div_factor=div_factor,
    final_div_factor=final_div_factor,
)

# Fast-forward to resume step
print(f"\nFast-forwarding scheduler to step {resume_step}...")
for _ in range(resume_step):
    scheduler.step()

current_lr = optimizer.param_groups[0]['lr']
print(f"Learning rate at resume (epoch {resume_epoch}, step {resume_step}): {current_lr:.8f}")

# Calculate initial and final learning rates
initial_lr = max_lr / div_factor
final_lr = max_lr / (div_factor * final_div_factor)
print(f"Initial LR (step 0): {max_lr:.10f}")
print(f"Final LR (step {total_steps}): {final_lr:.10f}")

# Simulate remaining epochs and show key points
print(f"\n{'Epoch':<8} {'Step':<10} {'Learning Rate':<20}")
print("-" * 40)

# Show current position
print(f"{resume_epoch:<8} {resume_step:<10} {current_lr:.10f} (resume)")

# Show checkpoints every 5 epochs
last_shown_epoch = resume_epoch
for epoch in range(resume_epoch + 5, num_epochs + 1, 5):
    step = epoch * iters_per_epoch
    steps_to_advance = (epoch - last_shown_epoch) * iters_per_epoch

    # Continue stepping from current position
    for _ in range(steps_to_advance):
        scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"{epoch:<8} {step:<10} {current_lr:.10f}")
    last_shown_epoch = epoch

# Show final epoch if not already shown
if last_shown_epoch < num_epochs:
    steps_to_advance = (num_epochs - last_shown_epoch) * iters_per_epoch - 1  # -1 because we haven't done the last step yet
    for _ in range(steps_to_advance):
        scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"{num_epochs:<8} {num_epochs * iters_per_epoch:<10} {current_lr:.10f} (final)")

print("\n" + "=" * 60)
print("✓ Schedule verification complete!")
print("=" * 60)
