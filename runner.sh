#!/bin/bash

source .venv/bin/activate
source .env

file-picker() {
  if [[ -z "$1" ]]; then
    echo "Usage: file-picker <glob_pattern>"
    echo "Example: file-picker data/logs/G1_mj_axis/*G1Cat_*"
    return 1
  fi
  shopt -s nullglob
  file_paths=($1)
  shopt -u nullglob
  file_names=("${file_paths[@]##*/}")

  if ((${#file_names[@]} == 0)); then
    echo "No matching entries found."
    return 1
  fi

  select file in "${file_names[@]}"; do
    if [[ -n $file ]]; then
      echo "$file"
      break
    else
      echo "Invalid selection"
    fi
  done
}

if [[ -n "$3" ]]; then
  task="$1"
  exp="$2"
  obs="$3"
else
  echo "Privileged tasks are more informative for distilling generalist policies."
  echo "Default tasks can be directly used for sim-to-real deployment."
  echo "Do you want privileged tasks? (y/N)"
  read -r ans
  ans=$(echo "$ans" | tr '[:upper:]' '[:lower:]')

  if [[ "$ans" == "y" ]]; then
    task="G1CatPri"
    exp=$(file-picker 'data/logs/G1_mj_axis/*G1CatPri_*')
  else
    task="G1Cat"
    exp=$(file-picker 'data/logs/G1_mj_axis/*G1Cat_*')
  fi
  echo "Random obstacles? (y/N): "
  read -r ans
  ans=$(echo "$ans" | tr '[:upper:]' '[:lower:]')
  if [[ "$ans" == "y" ]]; then
    obs=$(file-picker 'data/assets/RandObs/*')
  else
    obs=$(file-picker 'data/assets/TypiObs/*')
  fi
fi
echo "python -m cat_ppo.eval.mj_onnx_play --task \"$task\" --exp_name \"$exp\" --obs_name \"$obs\""
python -m cat_ppo.eval.mj_onnx_play --task "$task" --exp_name "$exp" --obs_name "$obs"
