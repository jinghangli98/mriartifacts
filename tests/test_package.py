import os
import subprocess
import sys

import mriartifacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_package_constants():
    assert mriartifacts.__version__ == "0.1.0"
    assert mriartifacts.FAMILIES == ("motion", "spike", "grappa", "lowres", "cs")
    assert mriartifacts.NULL_INDEX == 5 and mriartifacts.NUM_CLASSES == 6
    assert mriartifacts.PUBLIC_ARTIFACTS == ("motion", "spike", "lowres")
    assert set(mriartifacts.PUBLIC_ARTIFACTS) <= set(mriartifacts.FAMILIES)


def test_package_imports_standalone():
    """The package imports with only src/ on the path and from an unrelated
    working directory: no hidden dependency on files outside the package."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    subprocess.run([sys.executable, "-c", "import mriartifacts.cli, mriartifacts.export"],
                   check=True, env=env, cwd="/")
