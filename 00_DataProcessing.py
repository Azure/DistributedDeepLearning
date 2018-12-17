# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Data Processing
# In this notebook we convert the ImageNet data to the appropriate format so that we can use it for training.
#
# The dataset has many versions, the one commonly used for image classification is ILSVRC 2012. Go to the [download page](http://www.image-net.org/download-images) (you may need to register an account), and find the page for ILSVRC2012. You will need to download two files ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar

from pathlib import Path

DATA=Path("/data")

!mkdir -p {DATA/"train"}
!tar -C {DATA/"train"} -xf {DATA/"ILSVRC2012_img_train.tar"}

import tarfile
from tqdm import tqdm_notebook
import os

filenames = list((DATA/"train").glob("*.tar"))
pbar = tqdm_notebook(total=len(filenames))
for class_tar in filenames:
    pbar.set_description('Extracting '+class_tar.name+ '...')
    class_dir = os.path.splitext(class_tar)[0]
    os.mkdir(class_dir)
    with tarfile.open(class_tar) as f:
        f.extractall(class_dir)
    os.remove(class_tar)
    pbar.update(1)

!rm -r {DATA/"validation"}

!mkdir -p {DATA/"validation"}
!tar -C {DATA/"validation"} -xf {DATA/"ILSVRC2012_img_val.tar"}

# The validation data comes without labels so wee ned to run a script to asign the images to the appropriate classes.

validation_path = DATA/"validation"
validation_preparation_script = Path(os.getcwd())/"valprep.sh"

!bash -c "cd {validation_path} && {validation_preparation_script}"

# Finally we package the processed directories so that we can upload them quicker.

!cd {DATA} && tar -czvf train.tar.gz train

!cd {DATA} && tar -czvf validation.tar.gz validation
