#!/bin/bash
blender -b MagBallVis.blend --python vis_script.py -F FFMPEG -o //render_ -x .mp4 -a
