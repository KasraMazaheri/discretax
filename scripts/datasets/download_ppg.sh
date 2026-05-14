#!/usr/bin/env bash
# Download and extract the PPG-DaLiA dataset from the UCI ML Repository.
#
# Result layout (under data/raw/):
#   PPG_FieldStudy/S1/S1.pkl
#   PPG_FieldStudy/S2/S2.pkl
#   ...
#   PPG_FieldStudy/S15/S15.pkl
#
# Source: https://archive.ics.uci.edu/dataset/495/ppg+dalia
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
TARGET_DIR="${RAW_DIR}/PPG_FieldStudy"
URL="${PPG_URL:-https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip}"
OUTER_ARCHIVE="${RAW_DIR}/ppg_dalia.zip"

mkdir -p "${RAW_DIR}"

if [[ -d "${TARGET_DIR}" ]] && compgen -G "${TARGET_DIR}/S*/S*.pkl" > /dev/null; then
    echo "PPG-DaLiA already extracted at ${TARGET_DIR}"
    exit 0
fi

if [[ ! -f "${OUTER_ARCHIVE}" ]]; then
    echo "Downloading PPG-DaLiA archive from ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail -o "${OUTER_ARCHIVE}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${OUTER_ARCHIVE}" "${URL}"
    else
        echo "Error: neither curl nor wget is available" >&2
        exit 1
    fi
else
    echo "Outer archive already present at ${OUTER_ARCHIVE}"
fi

echo "Extracting outer archive into ${RAW_DIR}"
unzip -q -o "${OUTER_ARCHIVE}" -d "${RAW_DIR}"

# UCI nests the actual dataset inside an inner zip (data.zip / PPG_FieldStudy.zip).
# Extract any zip files produced by the outer extraction that look like PPG data.
while IFS= read -r inner_zip; do
    [[ -z "${inner_zip}" ]] && continue
    echo "Extracting inner archive ${inner_zip}"
    unzip -q -o "${inner_zip}" -d "${RAW_DIR}"
done < <(find "${RAW_DIR}" -maxdepth 2 -type f -name '*.zip' ! -path "${OUTER_ARCHIVE}")

if [[ ! -d "${TARGET_DIR}" ]]; then
    # Some mirrors extract into a differently-named top-level directory; try to relocate.
    candidate="$(find "${RAW_DIR}" -maxdepth 3 -type d -name 'PPG_FieldStudy' -print -quit || true)"
    if [[ -n "${candidate}" && "${candidate}" != "${TARGET_DIR}" ]]; then
        echo "Moving ${candidate} -> ${TARGET_DIR}"
        mv "${candidate}" "${TARGET_DIR}"
    fi
fi

if [[ ! -d "${TARGET_DIR}" ]] || ! compgen -G "${TARGET_DIR}/S*/S*.pkl" > /dev/null; then
    echo "Error: expected PPG subject pickles under ${TARGET_DIR}" >&2
    echo "Inspect ${RAW_DIR} to see what was extracted." >&2
    exit 1
fi

echo "Done. PPG-DaLiA raw data is at ${TARGET_DIR}"
