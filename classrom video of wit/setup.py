# -*- coding: utf-8 -*-
"""
SmartClass AI — Auto Setup & Run
==================================
Downloads the classroom video from Google Drive (if not already present),
then launches the analyzer automatically.

Usage:
    python setup.py          # download + run
    python setup.py --only-download  # only download, don't run
    python setup.py --only-run       # skip download, just run
"""
import sys, os, io, argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE         = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR    = os.path.join(HERE, "videos")
VIDEO_PATH   = os.path.join(VIDEO_DIR, "real_classroom.mp4")

# ── Google Drive file ID (extracted from the share link) ─────────────────────
# Share link: https://drive.google.com/file/d/1mMi55X5aMuWtf_ZrENXiBOLlGPIuRwvX/view
GDRIVE_FILE_ID = "1mMi55X5aMuWtf_ZrENXiBOLlGPIuRwvX"

# ── Expected file size (rough check to detect incomplete downloads) ───────────
MIN_SIZE_MB = 100   # if file is smaller than this, re-download


def download_video():
    """Download video from Google Drive using gdown or requests fallback."""
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # Check if already downloaded and complete
    if os.path.isfile(VIDEO_PATH):
        size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
        if size_mb >= MIN_SIZE_MB:
            print(f"  Video already downloaded ({size_mb:.0f} MB) — skipping.")
            return True
        else:
            print(f"  Incomplete download found ({size_mb:.1f} MB) — re-downloading...")
            os.remove(VIDEO_PATH)

    print("  Downloading classroom video from Google Drive...")
    print(f"  Destination: {VIDEO_PATH}")
    print()

    # ── Method 1: gdown (best for Google Drive) ───────────────────────────────
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, VIDEO_PATH, quiet=False, fuzzy=True)
        if os.path.isfile(VIDEO_PATH):
            size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
            if size_mb >= MIN_SIZE_MB:
                print(f"\n  Downloaded successfully ({size_mb:.0f} MB)")
                return True
            else:
                print(f"\n  Download incomplete ({size_mb:.1f} MB) — trying fallback...")
                os.remove(VIDEO_PATH)
    except ImportError:
        print("  gdown not installed — installing it now...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, VIDEO_PATH, quiet=False, fuzzy=True)
            if os.path.isfile(VIDEO_PATH):
                size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
                if size_mb >= MIN_SIZE_MB:
                    print(f"\n  Downloaded successfully ({size_mb:.0f} MB)")
                    return True
        except Exception as e:
            print(f"  gdown failed: {e}")
    except Exception as e:
        print(f"  gdown error: {e}")

    # ── Method 2: requests with streaming (fallback) ──────────────────────────
    print("  Trying requests fallback...")
    try:
        import requests
        # Google Drive direct download URL (bypasses the preview page)
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm=t"
        session = requests.Session()

        # First request — may get a virus scan warning page for large files
        resp = session.get(url, stream=True, timeout=30)

        # Handle Google's "file too large to scan" warning
        if "text/html" in resp.headers.get("Content-Type", ""):
            # Extract confirmation token from the warning page
            token = None
            for key, val in resp.cookies.items():
                if "download_warning" in key:
                    token = val
                    break
            if token is None:
                # Try parsing from response body
                for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
                    if "confirm=" in str(chunk):
                        import re
                        m = re.search(r'confirm=([0-9A-Za-z_\-]+)', str(chunk))
                        if m:
                            token = m.group(1)
                        break
            if token:
                url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}&confirm={token}"
                resp = session.get(url, stream=True, timeout=30)

        # Stream to file
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        bar_w = 30

        with open(VIDEO_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct    = downloaded / total * 100
                        filled = int(bar_w * pct / 100)
                        bar    = "#" * filled + "-" * (bar_w - filled)
                        mb     = downloaded / (1024 * 1024)
                        print(f"  [{bar}] {pct:5.1f}%  {mb:.0f} MB",
                              end="\r", flush=True)

        print()
        size_mb = os.path.getsize(VIDEO_PATH) / (1024 * 1024)
        if size_mb >= MIN_SIZE_MB:
            print(f"  Downloaded successfully ({size_mb:.0f} MB)")
            return True
        else:
            print(f"  Download may be incomplete ({size_mb:.1f} MB)")
            return False

    except Exception as e:
        print(f"  requests fallback failed: {e}")
        return False


def check_dependencies():
    """Check that Python packages are installed."""
    print("  Checking Python dependencies...")
    missing = []
    for pkg, import_name in [
        ("opencv-python", "cv2"),
        ("ultralytics",   "ultralytics"),
        ("mediapipe",     "mediapipe"),
        ("numpy",         "numpy"),
        ("Pillow",        "PIL"),
        ("colorama",      "colorama"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Installing...")
        req = os.path.join(os.path.dirname(HERE), "backend", "requirements.txt")
        if os.path.isfile(req):
            os.system(f"{sys.executable} -m pip install -r \"{req}\" -q")
        else:
            os.system(f"{sys.executable} -m pip install {' '.join(missing)} -q")
        print("  Done.\n")
    else:
        print("  All dependencies present.\n")


def run_analyzer():
    """Launch the analyzer script."""
    analyze = os.path.join(HERE, "analyze.py")
    print("\n" + "="*65)
    print("  Launching analyzer...")
    print("="*65 + "\n")
    os.execv(sys.executable, [sys.executable, analyze, VIDEO_PATH])


def main():
    parser = argparse.ArgumentParser(description="SmartClass AI — Setup & Run")
    parser.add_argument("--only-download", action="store_true",
                        help="Download video only, do not run analyzer")
    parser.add_argument("--only-run",      action="store_true",
                        help="Skip download, run analyzer immediately")
    args = parser.parse_args()

    print("="*65)
    print("  SmartClass AI — Auto Setup")
    print("="*65)

    if not args.only_run:
        check_dependencies()
        ok = download_video()
        if not ok and not args.only_download:
            print("\n  Download failed. You can manually place the video at:")
            print(f"  {VIDEO_PATH}")
            print("\n  Then run:  python analyze.py")
            sys.exit(1)
        if args.only_download:
            print("\n  Setup complete. Run 'python analyze.py' to start.")
            sys.exit(0)
    else:
        if not os.path.isfile(VIDEO_PATH):
            print(f"\n  ERROR: Video not found at:\n  {VIDEO_PATH}")
            print("  Run 'python setup.py' first to download it.")
            sys.exit(1)

    run_analyzer()


if __name__ == "__main__":
    main()
