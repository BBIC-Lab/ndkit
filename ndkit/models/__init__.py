import os
import glob
import importlib


def _auto_import_models():
    """
    Automatically import all model files ending with '*Model.py'.
    """

    current_dir = os.path.dirname(__file__)

    # Only load files ending with 'Model.py'
    model_files = glob.glob(os.path.join(current_dir, "*Model.py"))

    for file in model_files:
        name = os.path.basename(file)

        module_name = f"{__name__}.{name[:-3]}"  # strip .py

        try:
            importlib.import_module(module_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import model module '{module_name}': {e}"
            )


# Auto import when models package is loaded
_auto_import_models()
