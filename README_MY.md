# Kidney WSI segmentation of diagnostic tissue compartments

conda create --name milxseg python=3.8.15 -y
conda activate milxseg
pip3 install torch torchvision torchaudio (if on Moana, conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch)
pip install -U openmim
mim install mmcv-full==1.5.0 (may have some problem with mismatched cuda version, need to follow the 'INSTALLATION' part of this [page](https://stackoverflow.com/questions/39379792/install-cuda-without-root), need to choose cudatoolkit path that does not cause 'permission denied')
pip install mmdet
mim install mmengine (for installation verification)
pip install tifffile
pip install seaborn
pip install h5py
pip install scikit-learn
pip install kneed
pip install scikit-image
pip install slideio
pip install aicsimageio
pip install openslide-python
pip install wandb
pip install tqdm
pip install paquo

"ValueError: no valid qupath installation found"
Read the `Getting QuPath` section in the [Paquo document](https://pypi.org/project/paquo/)
Remember to do the export according to the output of `paquo get_qupath`, for example: `export PAQUO_QUPATH_DIR=/home/cougarnet.uh.edu/hqvo3/QuPath/QuPath-0.4.1`
