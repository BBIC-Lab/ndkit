import os
import glob
import importlib


def _auto_import_datasets():
    """
    Automatically import all dataset files ending with '*Dataset.py'.
    """

    current_dir = os.path.dirname(__file__)

    # Only load files ending with 'Dataset.py'
    dataset_files = glob.glob(os.path.join(current_dir, "*Dataset.py"))

    for file in dataset_files:
        name = os.path.basename(file)

        module_name = f"{__name__}.{name[:-3]}"  # strip .py

        try:
            importlib.import_module(module_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import dataset module '{module_name}': {e}"
            )


# Auto import when datasets package is loaded
_auto_import_datasets()
