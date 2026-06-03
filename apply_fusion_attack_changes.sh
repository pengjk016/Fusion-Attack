#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./apply_fusion_attack_changes.sh TARGET_OPENPILOT [options]

Copies Fusion-Attack changes into an already set up official openpilot tree.
Run the official clone/setup/build steps first, then run this script.

Options:
  --source SOURCE_OPENPILOT   Source openpilot tree. Default: ./openpilot next to this script.
  --patch-npy PATCH_FILE      Optional patch file to activate. It is copied to 1.npy and to the
                              .npy file loaded by source selfdrive/modeld/modeld.py.
  --lidar-source LIDAR_PY     Custom MetaDrive lidar.py source. Default:
                              /home/pjk/PycharmProjects/openpilot0.9.6/openpilot/.venv/lib/python3.11/site-packages/metadrive/component/sensors/lidar.py
  --skip-lidar                Do not replace MetaDrive's installed lidar.py.
  --no-backup                 Do not save .fusionattack-bak-* backups before overwriting.
  --dry-run                   Print planned copies without changing files.
  -h, --help                  Show this help.

Example:
  ./apply_fusion_attack_changes.sh /home/pjk/official/openpilot --patch-npy /path/to/1.npy
EOF
}

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_OPENPILOT="$SCRIPT_DIR/openpilot"
VISIONIPC_PYX_SOURCE="$SCRIPT_DIR/visionipc_pyx.so"
TARGET_OPENPILOT=""
PATCH_NPY=""
LIDAR_SOURCE="/home/pjk/PycharmProjects/openpilot0.9.6/openpilot/.venv/lib/python3.11/site-packages/metadrive/component/sensors/lidar.py"
SKIP_LIDAR=0
BACKUP=1
DRY_RUN=0
BACKUP_SUFFIX="fusionattack-bak-$(date +%Y%m%d-%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_OPENPILOT="${2:-}"
      shift 2
      ;;
    --patch-npy)
      PATCH_NPY="${2:-}"
      shift 2
      ;;
    --lidar-source)
      LIDAR_SOURCE="${2:-}"
      shift 2
      ;;
    --skip-lidar)
      SKIP_LIDAR=1
      shift
      ;;
    --no-backup)
      BACKUP=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$TARGET_OPENPILOT" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      TARGET_OPENPILOT="$1"
      shift
      ;;
  esac
done

if [[ -z "$TARGET_OPENPILOT" ]]; then
  echo "TARGET_OPENPILOT is required." >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$SOURCE_OPENPILOT" ]]; then
  echo "Source openpilot directory not found: $SOURCE_OPENPILOT" >&2
  exit 1
fi

if [[ ! -d "$TARGET_OPENPILOT" ]]; then
  echo "Target openpilot directory not found: $TARGET_OPENPILOT" >&2
  exit 1
fi

SOURCE_OPENPILOT="$(CDPATH= cd -- "$SOURCE_OPENPILOT" && pwd)"
TARGET_OPENPILOT="$(CDPATH= cd -- "$TARGET_OPENPILOT" && pwd)"

if [[ ! -f "$TARGET_OPENPILOT/pyproject.toml" || ! -f "$TARGET_OPENPILOT/SConstruct" || ! -d "$TARGET_OPENPILOT/selfdrive" ]]; then
  echo "Target does not look like an openpilot root: $TARGET_OPENPILOT" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_OPENPILOT/selfdrive/modeld/modeld.py" ]]; then
  echo "Source does not look like the Fusion-Attack openpilot tree: $SOURCE_OPENPILOT" >&2
  exit 1
fi

copy_file() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    echo "Missing source file: $src" >&2
    exit 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN copy: $src -> $dst"
    return
  fi

  mkdir -p "$(dirname -- "$dst")"

  if [[ "$BACKUP" -eq 1 && -f "$dst" ]]; then
    cp -p -- "$dst" "$dst.$BACKUP_SUFFIX"
  fi

  cp -p -- "$src" "$dst"
  echo "Copied: ${dst#$TARGET_OPENPILOT/}"
}

detect_active_patch_name() {
  python3 - "$SOURCE_OPENPILOT/selfdrive/modeld/modeld.py" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(r"np\.load\(\s*Path\(__file__\)\.parent\s*/\s*['\"]([^'\"]+\.npy)['\"]", text)
print(matches[0] if matches else "")
PY
}

find_target_lidar() {
  (
    cd "$TARGET_OPENPILOT"
    poetry run python - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("metadrive.component.sensors.lidar")
if spec is None or spec.origin is None:
  sys.exit(1)
print(spec.origin)
PY
  )
}

echo "Source: $SOURCE_OPENPILOT"
echo "Target: $TARGET_OPENPILOT"

RELATIVE_FILES=(
  "selfdrive/modeld/modeld.py"
  "selfdrive/controls/radard.py"
  "selfdrive/controls/dump_model_input_withradar_csv2.py"
  "tools/sim/bridge/metadrive/metadrive_bridge.py"
  "tools/sim/bridge/metadrive/metadrive_process.py"
  "tools/sim/bridge/metadrive/metadrive_world.py"
  "tools/sim/lib/simulated_car.py"
)

for rel in "${RELATIVE_FILES[@]}"; do
  copy_file "$SOURCE_OPENPILOT/$rel" "$TARGET_OPENPILOT/$rel"
done

copy_file "$VISIONIPC_PYX_SOURCE" "$TARGET_OPENPILOT/cereal/visionipc/visionipc_pyx.so"

for patch_rel in \
  "selfdrive/modeld/281.npy" \
  "selfdrive/modeld/optimpatch.npy" \
  "selfdrive/modeld/white.npy" \
  "selfdrive/modeld/random_gaussian_noise.npy" \
  "selfdrive/modeld/1.npy"; do
  if [[ -f "$SOURCE_OPENPILOT/$patch_rel" ]]; then
    copy_file "$SOURCE_OPENPILOT/$patch_rel" "$TARGET_OPENPILOT/$patch_rel"
  fi
done

if [[ -n "$PATCH_NPY" ]]; then
  if [[ ! -f "$PATCH_NPY" ]]; then
    echo "Patch file not found: $PATCH_NPY" >&2
    exit 1
  fi

  active_patch_name="$(detect_active_patch_name)"
  copy_file "$PATCH_NPY" "$TARGET_OPENPILOT/selfdrive/modeld/1.npy"

  if [[ -n "$active_patch_name" && "$active_patch_name" != "1.npy" ]]; then
    copy_file "$PATCH_NPY" "$TARGET_OPENPILOT/selfdrive/modeld/$active_patch_name"
    echo "Active modeld patch file detected from source modeld.py: $active_patch_name"
  fi
fi

if [[ "$SKIP_LIDAR" -eq 0 ]]; then
  if [[ ! -f "$LIDAR_SOURCE" ]]; then
    echo "MetaDrive lidar source not found: $LIDAR_SOURCE" >&2
    echo "Pass --lidar-source PATH or --skip-lidar." >&2
    exit 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN locate target MetaDrive lidar.py via: cd $TARGET_OPENPILOT && poetry run python ..."
    echo "DRY-RUN copy: $LIDAR_SOURCE -> <target metadrive/component/sensors/lidar.py>"
  else
    target_lidar="$(find_target_lidar || true)"
    if [[ -z "$target_lidar" ]]; then
      echo "Could not locate target MetaDrive lidar.py." >&2
      echo "Run the official setup first, then retry. Or pass --skip-lidar." >&2
      exit 1
    fi
    copy_file "$LIDAR_SOURCE" "$target_lidar"
    echo "Replaced MetaDrive lidar.py: $target_lidar"
  fi
fi

echo
echo "Fusion-Attack changes applied."
if [[ "$BACKUP" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  echo "Backups have suffix: .$BACKUP_SUFFIX"
fi
echo "If the official tree was already built, these Python/.npy replacements do not require a full rebuild."
