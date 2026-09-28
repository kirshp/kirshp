#!/usr/bin/env bash
# Создаёт виртуальное окружение .venv и ставит зависимости.
#   ./setup.sh        — CPU-версия PyTorch (работает везде, ~200 МБ)
#   ./setup.sh gpu    — сборка PyTorch с CUDA (нужна видеокарта NVIDIA)
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if [[ "${1:-cpu}" == "gpu" ]]; then
  pip install torch numpy
else
  pip install torch numpy --index-url https://download.pytorch.org/whl/cpu
fi

python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "Готово. Активируйте окружение: source .venv/bin/activate"
