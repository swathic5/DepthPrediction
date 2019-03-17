# DepthPrediction
A tool to predict the depth field of a 2-dimensional image based on 
"Deep convolutional neural fields for depth estimation from a single image" by Liu et al.
which can be found [here on arViX](http://arxiv.org/abs/1411.6387).

The ipython notebooks for the project can be viewed [here](http://nbviewer.ipython.org/github/asousa/DepthPrediction/tree/master/).

Install Caffe
    - Cloned 
    - Update Makefile and Makefile.config
    - Do make and build pycaffe
    - Copy Caffe directory under python to /usr/lib/python2.7/dist-packages
    - Update LD_LIBRARY_PATH by adding a new path to point to built caffe library
Fixed indentation errors
Comment rc_params fontsize
Handle type of save_fig.dpi (assign ball park float value)
Restart kernel after each run to fix auto_reload problem
Install and correct gco_python package

Download NYU labelled dataset - https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html
Download test/train split - https://cs.nyu.edu/~silberman/projects/indoor_scene_seg_sup.html
Updated hardcoded paths for training data, train/test split and output in build_dataset_directory.py
Create output directories
Run build_dataset_directory.py



