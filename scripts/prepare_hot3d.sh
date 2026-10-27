#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HAWOR_PYTHON="${HAWOR_PYTHON:-/home/cjt/.conda/envs/hawor/bin/python}"
if [[ ! -x "${HAWOR_PYTHON}" ]]; then
    HAWOR_PYTHON="$(command -v python)"
fi

export PYTHONNOUSERSITE=1
export PATH="$(dirname "${HAWOR_PYTHON}"):${PATH}"

DOWNLOADER_DIR="${ROOT_DIR}/hot3d/data_downloader"
ARIA_MANIFEST="${DOWNLOADER_DIR}/Hot3DAria_download_urls.json"
ASSETS_MANIFEST="${DOWNLOADER_DIR}/Hot3DAssets_download_urls.json"
DATASET_ROOT="${ROOT_DIR}/hot3d/dataset"
EXPORT_ROOT="${ROOT_DIR}/datasets/hot3d_valset_export"
MANO_ROOT="${ROOT_DIR}/hot3d/mano_v1_2/models"

SEQUENCES=(
    P0001_a68492d5 P0001_9b6feab7 P0014_8254f925
    P0011_76ea6d47 P0014_84ea2dcc P0001_8d136980
    P0012_476bae57 P0012_130a66e1 P0014_24cb3bf0
    P0010_1c9fe708 P0002_2ea9af5b P0011_11475e24
    P0010_0ecbf39f P0010_160e551c P0015_42b8b389
    P0012_915e71c6 P0002_65085bfc P0011_47878e48
    P0011_cee8fe4f P0002_016222d1 P0012_d85e10f6
    P0012_119de519 P0010_41c4c626 P0012_f7e3880b
    P0009_02511c2f P0011_72efb935 P0010_924e574e
)

require_file() {
    if [[ ! -s "$1" ]]; then
        echo "Missing required file: $1" >&2
        return 1
    fi
}

check_common() {
    require_file "${MANO_ROOT}/MANO_LEFT.pkl"
    require_file "${MANO_ROOT}/MANO_RIGHT.pkl"
    "${HAWOR_PYTHON}" -c \
        "import cv2, joblib, projectaria_tools, pyvrs, rerun, torch; print('HOT3D Python imports: OK')"
    command -v ffmpeg >/dev/null
    echo "MANO models: OK"
    echo "ffmpeg: $(command -v ffmpeg)"
    df -h "${ROOT_DIR}" | tail -n 1
}

check_manifests() {
    local missing=0
    for manifest in "${ARIA_MANIFEST}" "${ASSETS_MANIFEST}"; do
        if [[ ! -s "${manifest}" ]]; then
            echo "Missing required file: ${manifest}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        return 1
    fi
    "${HAWOR_PYTHON}" -c \
        "import json; [json.load(open(p)) for p in ('$ARIA_MANIFEST', '$ASSETS_MANIFEST')]; print('HOT3D manifests: valid JSON')"
    "${HAWOR_PYTHON}" -c \
        "import json,sys; available=set(json.load(open(sys.argv[1]))['sequences']); missing=[s for s in sys.argv[2:] if s not in available]; assert not missing, f'Missing requested sequences in Aria manifest: {missing}'; print(f'Aria manifest contains all {len(sys.argv)-2} requested sequences')" \
        "${ARIA_MANIFEST}" "${SEQUENCES[@]}"
}

download_hot3d() {
    check_common
    check_manifests
    mkdir -p "${DATASET_ROOT}"
    (
        cd "${DOWNLOADER_DIR}"
        "${HAWOR_PYTHON}" dataset_downloader_base_main.py \
            --cdn_file "${ASSETS_MANIFEST}" \
            --output_folder "${DATASET_ROOT}" \
            --sequence_names all
        "${HAWOR_PYTHON}" dataset_downloader_base_main.py \
            --cdn_file "${ARIA_MANIFEST}" \
            --output_folder "${DATASET_ROOT}" \
            --data_types all \
            --sequence_names "${SEQUENCES[@]}"
    )
}

export_labels() {
    check_common
    require_file "${DATASET_ROOT}/assets/instance.json"
    mkdir -p "${EXPORT_ROOT}"
    (
        cd "${ROOT_DIR}/hot3d"
        "${HAWOR_PYTHON}" export_gt.py \
            --dataset-root "${DATASET_ROOT}" \
            --output-root "${EXPORT_ROOT}" \
            --start-frame 20 \
            --resume
    )
}

preprocess_hot3d() {
    check_common
    require_file "${EXPORT_ROOT}/val.json"
    (
        cd "${ROOT_DIR}"
        "${HAWOR_PYTHON}" lib/datasets/hot3d_dataset_preprocess.py \
            --video_root "${EXPORT_ROOT}" \
            --set_file val.json \
            --for_eval
    )
}

usage() {
    echo "Usage: $0 {check|download|export|preprocess|all}"
}

stage="${1:-check}"
case "${stage}" in
    check)
        check_common
        check_manifests
        ;;
    download)
        download_hot3d
        ;;
    export)
        export_labels
        ;;
    preprocess)
        preprocess_hot3d
        ;;
    all)
        download_hot3d
        export_labels
        preprocess_hot3d
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
