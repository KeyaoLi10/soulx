script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname "$(dirname "$script_dir")")

cd $root_dir
uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1