import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
from viewer_export_gt import export_gt
sequence=sys.argv[1]
output=ROOT/'datasets/hot3d_valset_export'/sequence
required=['anno.pth','head_pose.pkl','ego_extrinsics.pkl','focal.txt','tt_list.pkl',f'{sequence}.mp4']
if not all((output/x).is_file() and (output/x).stat().st_size>0 for x in required):
 export_gt(str(ROOT/'hot3d/dataset'/sequence),start_frame=20,output_folder=str(output),object_library_folder=str(ROOT/'hot3d/dataset/assets'),mano_model_folder=str(ROOT/'hot3d/mano_v1_2/models'))
print(f'EXPORT_COMPLETED {sequence}',flush=True)
