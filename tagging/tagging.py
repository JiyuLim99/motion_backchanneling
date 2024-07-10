import logging
import os
import pickle
import numpy as np
import pandas as pd
from fairmotion.fairmotion.data import bvh
from fairmotion.fairmotion.ops import motion as motion_ops
from fairmotion.fairmotion.ops import conversions
from fairmotion.fairmotion.utils import utils as fairmotion_utils

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def split_into_windows(motion, window_size, stride):
    """
    Split motion object into list of motions with length window_size with
    the given stride.
    """
    n_windows = (motion.num_frames() - window_size) // stride + 1
    motion_ws = []

    for start in stride * np.arange(n_windows):
        end = start + window_size
        window = motion_ops.cut(motion, start, end)
        motion_ws.append((window, (start, end)))

    return motion_ws

def save_as_pickle(motion_windows, output_path):
    convert_fn = conversions.R2Q
    data = [(convert_fn(motion.rotations()), indices, types) for motion, indices, types in motion_windows]
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    logging.info(f"Saved {output_path}")

def find_start_frames(df_filtered, ends, start_frames, frame_range, window_stride):
    start_indices = []
    for index, start_frame in enumerate(start_frames):
        start_diffs = [(end - start_frame) for end in ends]
        indices_to_add = [i for i, diff in enumerate(start_diffs) if frame_range <= diff < frame_range + window_stride]
        start_indices.extend(indices_to_add)
    return start_indices.copy() if start_indices else None

def find_end_frames(df_filtered, starts, end_frames, frame_range, window_stride):
    end_indices = []
    for index, end_frame in enumerate(end_frames):
        end_diffs = [(end_frame - start) for start in starts]
        indices_to_add = [i for i, diff in enumerate(end_diffs) if frame_range <= diff < frame_range + window_stride]
        end_indices.extend(indices_to_add)
    return end_indices if end_indices else None

def process_bvh_files(input_dir, output_dir, window_size, window_stride, csv_path):
    fairmotion_utils.create_dir_if_absent(output_dir)

    df = pd.read_csv(csv_path, encoding_errors='ignore')

    for filepath in fairmotion_utils.files_in_dir(input_dir, ext="bvh"):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        motion = bvh.load(filepath)
        motion_windows = list(split_into_windows(motion, window_size, window_stride))

        df_filtered = df[df['Filename'] == filename]
        start_frames = df_filtered['Start frame'].tolist()
        end_frames = df_filtered['End frame'].tolist()
        types = df_filtered['Type'].tolist()

        annotated_windows = []
        starts = [start for window, (start, end) in motion_windows]
        ends = [end for window, (start, end) in motion_windows]

        if not df_filtered.empty:
            start_indices = find_start_frames(df_filtered, ends, start_frames, frame_range, window_stride)
            end_indices = find_end_frames(df_filtered, starts, end_frames, frame_range, window_stride)
               
            i = 0
            while i < len(motion_windows):
                if i in start_indices:
                    index_in_start_indices = start_indices.index(i)
                    end_indices_frame = end_indices[index_in_start_indices]
                    types_value = int(types[index_in_start_indices])

                    for j in range(i, end_indices_frame + 1):
                        window, (start, end) = motion_windows[j]
                        annotated_windows.append((window, (start, end), types_value))
                    i = end_indices_frame + 1
                else:
                    window, (start, end) = motion_windows[i]
                    annotated_windows.append((window, (start, end), -1))
                    i += 1                
        else:
            for window, (start, end) in motion_windows:
                annotated_windows.append((window, (start, end), -1))

        output_path = os.path.join(output_dir, f"{filename}.pkl")
        save_as_pickle(annotated_windows, output_path)


if __name__ == "__main__":
    input_dir = 'C:/Users/etri/Desktop/etri/data/trn_data'
    output_dir = 'C:/Users/etri/Desktop/etri/data/new_data'
    csv_path = 'C:/Users/etri/Desktop/etri/data/Backchanelling Annotation.csv' 
    window_size = 30
    window_stride = 15
    frame_range = 20

    process_bvh_files(input_dir, output_dir, window_size, window_stride, csv_path)
