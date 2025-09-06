set -e

# --- Find Site-Packages of a Pre-installed Library ---
SITE_PACKAGES_PATH=$(python -c '
import os, importlib.util
try:
    spec = importlib.util.find_spec("torch")

    if spec and spec.origin and "__init__.py" in spec.origin:
        print(os.path.dirname(os.path.dirname(spec.origin)))
except Exception:
    pass
')

# --- Conditionally Set PYTHONPATH ---
if [ -n "$SITE_PACKAGES_PATH" ]; then
    echo "export PYTHONPATH=\"${SITE_PACKAGES_PATH}:\${PYTHONPATH}\"" >> ~/.bashrc
    echo "PYTHONPATH=${SITE_PACKAGES_PATH}" > .devcontainer/.pylance.env
else
    echo "No pre-installed packages found to inherit from."
fi