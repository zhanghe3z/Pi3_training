#!/usr/bin/env python3
"""
Test script to verify TartanAir Hospital dataset loading works correctly.
Run this before starting training to ensure everything is configured properly.
"""

import sys
sys.path.append('.')

import numpy as np
from datasets.tartanair_hospital_dataset import TarTanAirHospitalDataset
from datasets.base.transforms import JitterJpegLossBlurring, ImgToTensor

def test_dataset():
    print("=" * 60)
    print("Testing TartanAir Hospital Dataset Loading")
    print("=" * 60)
    print()

    # Test training dataset
    print("[1/2] Testing Training Dataset...")
    train_dataset = TarTanAirHospitalDataset(
        data_root='/mnt/localssd/tartanair_tools/tartanair_data/hospital',
        z_far=80,
        frame_num=8,
        resolution=[[518, 336]],
        aug_crop=16,
        transform=JitterJpegLossBlurring,
        aug_focal=0.9,
        mode='train',
        verbose=True
    )

    print(f"Number of sequences: {len(train_dataset)}")
    print()

    # Try to load one sample
    print("Loading first training sample...")
    try:
        sample = train_dataset[0]
        print(f"✓ Successfully loaded sample!")
        print(f"  - Number of views: {len(sample['views'])}")
        print(f"  - Image shape: {sample['views'][0]['img'].shape}")
        print(f"  - Depth shape: {sample['views'][0]['depthmap'].shape}")
        print(f"  - Camera pose shape: {sample['views'][0]['camera_pose'].shape}")
        print(f"  - Camera intrinsics shape: {sample['views'][0]['camera_intrinsics'].shape}")
        print(f"  - Dataset label: {sample['views'][0]['dataset']}")
        print(f"  - Scene label: {sample['views'][0]['label']}")
    except Exception as e:
        print(f"✗ Failed to load sample!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test validation dataset
    print("[2/2] Testing Test/Validation Dataset...")
    test_dataset = TarTanAirHospitalDataset(
        data_root='/mnt/localssd/tartanair_tools/tartanair_data/hospital',
        z_far=80,
        frame_num=8,
        resolution=[[518, 336]],
        transform=ImgToTensor,
        mode='test',
        verbose=True
    )

    print(f"Number of sequences: {len(test_dataset)}")
    print()

    # Try to load one sample
    print("Loading first test sample...")
    try:
        sample = test_dataset[0]
        print(f"✓ Successfully loaded sample!")
        print(f"  - Number of views: {len(sample['views'])}")
        print(f"  - Image shape: {sample['views'][0]['img'].shape}")
        print(f"  - Depth shape: {sample['views'][0]['depthmap'].shape}")
    except Exception as e:
        print(f"✗ Failed to load sample!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("Dataset Test Completed Successfully!")
    print("=" * 60)
    print()
    print("You can now run training with:")
    print("  ./train_hospital_simple.sh 1")
    print("or")
    print("  ./train_hospital_from_scratch.sh")
    print()

    return True

if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)
