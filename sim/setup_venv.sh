#!/bin/bash
python3 -m venv ~/arms-venv
~/arms-venv/bin/pip install -q --upgrade pip
~/arms-venv/bin/pip install -q numpy matplotlib pillow opencv-python-headless
~/arms-venv/bin/python -c "import tkinter, matplotlib, numpy, cv2; from PIL import ImageTk; print('ok tk',tkinter.TkVersion,'mpl',matplotlib.__version__,'np',numpy.__version__,'cv2',cv2.__version__)"
