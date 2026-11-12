import sys

from deeplabcut import analyze_videos
from deeplabcut import video_inference_superanimal
import time
import gc
import glob
import json
import threading
import ctypes
import os
from datetime import datetime

# Written into the destination folder after a successful run. The folder is named for
# the model's Task — the short name you chose in its config and the one you quote to
# ask for these coordinates later. This records the rest: the full model name
# DeepLabCut generated, which config produced it, and when. So the exact snapshot is a
# recorded fact rather than something inferred from a filename.
MODEL_RECORD = 'model.json'

# Global flag for malloc_trim thread
stop_trim_thread = False


def scorer_name(dest_folder, returned):
    """
    The full model name DeepLabCut analysed with.

    Prefers what analyze_videos returned; falls back to reading it back off the file
    that was just written, since the name is embedded there either way.
    """
    if isinstance(returned, str) and returned:
        return returned

    for path in sorted(glob.glob(os.path.join(dest_folder, '*_videoDLC_*'))):
        stem = os.path.basename(path).rsplit('.', 1)[0]
        return stem.split('_videoDLC_', 1)[1]
    return None


def write_model_record(dest_folder, config_path, video_path, scorer):
    """Record what produced the coordinates now sitting in dest_folder."""
    record = {
        'task': os.path.basename(os.path.normpath(dest_folder)),
        'model': scorer,
        'config': str(config_path),
        'video': str(video_path),
        'analysed': datetime.now().isoformat(timespec='seconds'),
    }
    path = os.path.join(dest_folder, MODEL_RECORD)
    with open(path, 'w') as handle:
        json.dump(record, handle, indent=2)
    print(f"Wrote {path}")
    return record


def malloc_trim_thread(interval=15):
    """
    Periodically call malloc_trim to return freed memory to the OS.
    This prevents memory fragmentation issues during long video analysis.
    """
    global stop_trim_thread

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        print("Warning: malloc_trim not available (Linux only). Memory usage may be higher.")
        return

    while not stop_trim_thread:
        time.sleep(interval)
        if not stop_trim_thread:
            # Trim memory and run garbage collection
            libc.malloc_trim(0)
            gc.collect()


def analyse(video_path, gpu_id, config_path, dest_folder=None):
    """
    Analyse video with malloc_trim fix for memory issues.

    Args:
        video_path: Path to the video file
        gpu_id: GPU ID to use for analysis
        config_path: Path to the DLC model's config.yaml
        dest_folder: Where to write the coordinates. Without it DeepLabCut writes
            beside the video, which in a nested session means writing into the raw
            acquisition folder — so the pipeline always passes this.
    """
    global stop_trim_thread
    stop_trim_thread = False

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Using model config: {config_path}")
    if dest_folder:
        os.makedirs(dest_folder, exist_ok=True)
        print(f"Writing coordinates to: {dest_folder}")

    # Start malloc_trim thread
    trim_thread = threading.Thread(target=malloc_trim_thread, args=(15,), daemon=True)
    trim_thread.start()

    start_time = time.perf_counter()
    print(f"Analyzing {str(video_path)}")
    print(f"Using GPU {gpu_id}")

    try:
        returned = analyze_videos(config=config_path,
                                  videos=[video_path],
                                  # DeepLabCut filters the given videos by this
                                  # extension, so it must be the video's own: a
                                  # hardcoded '.avi' silently skips .mkv and .mp4.
                                  videotype=os.path.splitext(video_path)[1].lower() or '.avi',
                                  save_as_csv=True,
                                  gputouse=gpu_id,
                                  destfolder=dest_folder)

        if dest_folder:
            write_model_record(dest_folder, config_path, video_path,
                               scorer_name(dest_folder, returned))

        # Calculate time taken
        elapsed_time = time.perf_counter() - start_time
        minutes = round(elapsed_time / 60, 2)
        seconds = round(elapsed_time % 60, 2)

        print(f"Analysis of {str(video_path)} complete.")
        print(f"Took: {minutes} minutes and {seconds} seconds")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    finally:
        # Stop malloc_trim thread
        stop_trim_thread = True


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python batch_analyse.py VIDEO_PATH GPU_ID CONFIG_PATH [DEST_FOLDER]")
        print("Example: python batch_analyse.py /path/to/video.avi 0 /path/to/config.yaml "
              "/path/to/session/dlc/my_task")
        sys.exit(1)

    video_path = sys.argv[1]
    gpu_id = int(sys.argv[2])
    config_path = sys.argv[3]
    dest_folder = sys.argv[4] if len(sys.argv) > 4 else None

    analyse(video_path, gpu_id, config_path, dest_folder)