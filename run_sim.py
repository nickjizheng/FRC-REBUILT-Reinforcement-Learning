import os
from pathlib import Path
import sys

# Non-interactive acceptance of the NVIDIA Omniverse Kit EULA (already accepted
# for this install).  Without this, launching via LaunchSimulator.exe blocks on the
# "Do you accept the EULA? (Yes/No):" prompt with no console to answer it, so the
# sim appears to never start.  setdefault lets an external override still win.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from xrc_rebuilt.isaac_scene import main


if __name__ == "__main__":
    main()
