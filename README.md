# Ball tracking in volleyball

This repo is for ball recognition and tracking in live volleyball game.

## Requirements
- Python3
- OpenCV
- Keras with Tensorflow

## How to use

1. Get a video file with a game fragment
2. Get highest blobs:

*python3 high.py <path_to_vb_file> <mask_output_dir> <color_output_dir>*

3. Classify manually the blobs into 2 classes (b)all/(n)ot ball
4. Put the classified data into vball-net/train
5. cd vball-net
6. Python3 train.py
7. Test blobber: 

*python3 blobber.py <path_to_vb_file> 

You will see and output with ball paths like that:

<img src="https://github.com/tprlab/vball/raw/master/images/paths.jpg"/>

8. Run a player with the live ball tracking:

*python3 ball_play.py <path_to_vb_file> 

<img src="https://github.com/tprlab/vball/raw/master/images/vball_tracking.gif"/>

## Links

- Used an open dataset from [some austrian league](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/vb14/)
- [Story @ TowardDataScience](https://towardsdatascience.com/ball-tracking-in-volleyball-with-opencv-and-tensorflow-3d6e857bd2e7)
- [Story @ Habr(Russian)](https://habr.com/ru/post/505672/)
- [vball.io](https://vball.io) - a service I started to cut rallies and digest volleball videos based on this ball tracking algo.

