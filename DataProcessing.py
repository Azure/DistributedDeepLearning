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

from pathlib import Path

DATA=Path("/data/imagenet")

!rm -r {DATA / "train.tar.gz"}

!ls {DATA}

!mkdir -p {DATA/"train"}
!tar -C {DATA/"train"} -xf {DATA/"ILSVRC2012_img_train.tar"}

!pip install tqdm

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

validation_path = DATA/"validation"
validation_preparation_script = Path(os.getcwd())/"valprep.sh"

!bash -c "cd {validation_path} && {validation_preparation_script}"

!cd {DATA} && tar -czvf train.tar.gz train

!cd {DATA} && tar -czvf validation.tar.gz validation

SUBSCRIPTION="Boston Team Danielle"

!az login -o table

!az account set -s ${SUBSCRIPTION}
