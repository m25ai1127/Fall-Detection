"""
Dataset Download Utility
========================
Downloads the UR Fall Detection (URFD) dataset for training and evaluation.
Also provides instructions for NTU RGB+D dataset access.
"""

import os
import sys
import zipfile
import urllib.request
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


# URFD dataset file URLs (camera 0 RGB and depth sequences)
URFD_BASE_URL = "http://fenix.univ.rzeszow.pl/~mkepski/ds/data"

URFD_FALL_FILES = {
    f"fall-{i:02d}-cam0-rgb": f"{URFD_BASE_URL}/fall-{i:02d}-cam0-rgb.zip"
    for i in range(1, 31)
}
URFD_FALL_DEPTH_FILES = {
    f"fall-{i:02d}-cam0-d": f"{URFD_BASE_URL}/fall-{i:02d}-cam0-d.zip"
    for i in range(1, 31)
}
URFD_ADL_FILES = {
    f"adl-{i:02d}-cam0-rgb": f"{URFD_BASE_URL}/adl-{i:02d}-cam0-rgb.zip"
    for i in range(1, 41)
}
URFD_ADL_DEPTH_FILES = {
    f"adl-{i:02d}-cam0-d": f"{URFD_BASE_URL}/adl-{i:02d}-cam0-d.zip"
    for i in range(1, 41)
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc="Downloading"):
    """Download a file with progress bar."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"  [SKIP] Already exists: {output_path.name}")
        return True
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, str(output_path), reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(extract_to))
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to extract {zip_path}: {e}")
        return False


def download_urfd(max_sequences=None):
    """
    Download and extract URFD dataset.
    
    Args:
        max_sequences: Limit number of sequences to download (for testing)
    """
    print("=" * 60)
    print("UR Fall Detection Dataset (URFD) Download")
    print("=" * 60)
    print(f"Source: {config.URFD_DATASET_URL}")
    print(f"Output: {config.URFD_RAW_DIR}")
    print()
    
    config.URFD_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare download lists
    all_files = {}
    all_files.update(URFD_FALL_FILES)
    all_files.update(URFD_FALL_DEPTH_FILES)
    all_files.update(URFD_ADL_FILES)
    all_files.update(URFD_ADL_DEPTH_FILES)
    
    if max_sequences:
        # Limit to first N fall + N ADL sequences (with RGB + depth)
        limited = {}
        for i in range(1, min(max_sequences + 1, 31)):
            key_rgb = f"fall-{i:02d}-cam0-rgb"
            key_d = f"fall-{i:02d}-cam0-d"
            if key_rgb in URFD_FALL_FILES:
                limited[key_rgb] = URFD_FALL_FILES[key_rgb]
            if key_d in URFD_FALL_DEPTH_FILES:
                limited[key_d] = URFD_FALL_DEPTH_FILES[key_d]
        for i in range(1, min(max_sequences + 1, 41)):
            key_rgb = f"adl-{i:02d}-cam0-rgb"
            key_d = f"adl-{i:02d}-cam0-d"
            if key_rgb in URFD_ADL_FILES:
                limited[key_rgb] = URFD_ADL_FILES[key_rgb]
            if key_d in URFD_ADL_DEPTH_FILES:
                limited[key_d] = URFD_ADL_DEPTH_FILES[key_d]
        all_files = limited
    
    total = len(all_files)
    success = 0
    
    print(f"Downloading {total} files...")
    print()
    
    for name, url in all_files.items():
        zip_path = config.URFD_RAW_DIR / f"{name}.zip"
        extract_dir = config.URFD_RAW_DIR / name
        
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  [SKIP] Already extracted: {name}")
            success += 1
            continue
        
        print(f"  [{success + 1}/{total}] {name}")
        if download_file(url, zip_path, desc=f"  {name}"):
            if extract_zip(zip_path, config.URFD_RAW_DIR):
                # Clean up zip file to save space
                zip_path.unlink(missing_ok=True)
                success += 1
    
    print()
    print(f"Download complete: {success}/{total} files")
    print(f"Data saved to: {config.URFD_RAW_DIR}")
    
    return success == total


def print_ntu_instructions():
    """Print instructions for downloading NTU RGB+D dataset."""
    print()
    print("=" * 60)
    print("NTU RGB+D Dataset (Supplementary)")
    print("=" * 60)
    print()
    print("The NTU RGB+D dataset requires registration for download.")
    print()
    print("Steps:")
    print("  1. Visit: https://rose1.ntu.edu.sg/dataset/actionRecognition/")
    print("  2. Register for an account")
    print("  3. Request access to NTU RGB+D dataset")
    print("  4. Download the RGB videos and depth map files")
    print("  5. Extract relevant action classes:")
    print(f"     - A{config.NTU_FALL_ACTION_ID:03d}: Falling down")
    print(f"     - Normal actions: {['A' + str(i).zfill(3) for i in config.NTU_NORMAL_ACTION_IDS]}")
    print(f"  6. Place files in: {config.NTU_RAW_DIR}")
    print()
    print("Note: NTU RGB+D is supplementary. The project works with URFD alone.")


def verify_dataset():
    """Verify downloaded dataset structure."""
    print()
    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    
    # Check URFD
    urfd_falls = 0
    urfd_adls = 0
    
    if config.URFD_RAW_DIR.exists():
        for d in config.URFD_RAW_DIR.iterdir():
            if d.is_dir():
                if "fall" in d.name:
                    urfd_falls += 1
                elif "adl" in d.name:
                    urfd_adls += 1
    
    print(f"\nURFD Dataset:")
    print(f"  Fall sequences: {urfd_falls // 2} (RGB+Depth pairs)")
    print(f"  ADL sequences:  {urfd_adls // 2} (RGB+Depth pairs)")
    print(f"  Location: {config.URFD_RAW_DIR}")
    
    if urfd_falls > 0 or urfd_adls > 0:
        print("  Status: [OK] Data found")
    else:
        print("  Status: [X] No data found - run download first")
    
    return urfd_falls > 0


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for fall detection"
    )
    parser.add_argument(
        "--dataset", type=str, default="urfd",
        choices=["urfd", "ntu", "all"],
        help="Dataset to download (default: urfd)"
    )
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Maximum number of sequences to download (for testing)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing dataset without downloading"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_dataset()
        return
    
    if args.dataset in ["urfd", "all"]:
        download_urfd(max_sequences=args.max_sequences)
    
    if args.dataset in ["ntu", "all"]:
        print_ntu_instructions()
    
    verify_dataset()


if __name__ == "__main__":
    main()
