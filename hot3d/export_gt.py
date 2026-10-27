import argparse
import pickle
from viewer_export_gt import export_gt
import json
import os
# import concurrent.futures


HOT3D_VAL_SEQUENCES = [
    'P0001_a68492d5', 'P0001_9b6feab7', 'P0014_8254f925',
    'P0011_76ea6d47', 'P0014_84ea2dcc', 'P0001_8d136980',
    'P0012_476bae57', 'P0012_130a66e1', 'P0014_24cb3bf0',
    'P0010_1c9fe708', 'P0002_2ea9af5b', 'P0011_11475e24',
    'P0010_0ecbf39f', 'P0010_160e551c', 'P0015_42b8b389',
    'P0012_915e71c6', 'P0002_65085bfc', 'P0011_47878e48',
    'P0011_cee8fe4f', 'P0002_016222d1', 'P0012_d85e10f6',
    'P0012_119de519', 'P0010_41c4c626', 'P0012_f7e3880b',
    'P0009_02511c2f', 'P0011_72efb935', 'P0010_924e574e',
]


def load_set(set_file='train_set.pkl'):
    try:
        with open(set_file, 'rb') as f:
            set = pickle.load(f)
        return set
    except FileNotFoundError:
        return f"Error: The file {set_file} does not exist."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='dataset')
    parser.add_argument('--output-root', default='hot3d_dataset_export')
    parser.add_argument('--start-frame', type=int, default=20)
    parser.add_argument('--sequences', nargs='+', default=HOT3D_VAL_SEQUENCES)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    sequence_set = args.sequences
    os.makedirs(args.output_root, exist_ok=True)
    print(sequence_set)
    cnt = 0
    for video_name in sequence_set:
        sequence_folder = os.path.join(args.dataset_root, video_name)
        output_folder = os.path.join(args.output_root, video_name)
        required_outputs = [
            os.path.join(output_folder, 'anno.pth'),
            os.path.join(output_folder, 'head_pose.pkl'),
            os.path.join(output_folder, 'ego_extrinsics.pkl'),
            os.path.join(output_folder, 'focal.txt'),
            os.path.join(output_folder, 'tt_list.pkl'),
            os.path.join(output_folder, f'{video_name}.mp4'),
        ]
        if args.resume and all(os.path.exists(path) for path in required_outputs):
            cnt += 1
            print(f'skip completed sequence {video_name} ({cnt}/{len(sequence_set)})')
            continue
        if not os.path.isdir(sequence_folder):
            raise FileNotFoundError(f'Missing HOT3D sequence folder: {sequence_folder}')
        export_gt(
            sequence_folder,
            start_frame=args.start_frame,
            output_folder=output_folder,
            object_library_folder=os.path.join(args.dataset_root, 'assets'),
            mano_model_folder=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'mano_v1_2', 'models'
            ),
        )
        cnt += 1
        print(f"finished {cnt}/{len(sequence_set)} test videos")
    with open(os.path.join(args.output_root, 'val.json'), 'w') as f:
        json.dump(sequence_set, f)
