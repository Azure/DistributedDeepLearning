#!/usr/bin/env bash
wget https://gist.githubusercontent.com/msalvaris/073c28a9993d58498957294d20d74202/raw/87a78275879f7c9bb8d6fb9de8a2d2996bb66c24/install_azcopy
chmod 777 install_azcopy
sudo ./install_azcopy

mkdir -p /data/imagenet
azcopy --source https://datasharesa.blob.core.windows.net/imagenet/validation.csv \
        --destination  /data/imagenet/validation.csv\
        --source-sas "?se=2025-01-01&sp=r&sv=2017-04-17&sr=b&sig=7x3rN7c/nlXbnZ0gAFywd5Er3r6MdwCq97Vwvda25WE%3D"\
        --quiet

azcopy --source https://datasharesa.blob.core.windows.net/imagenet/validation.tar.gz \
        --destination  /data/imagenet/validation.tar.gz\
        --source-sas "?se=2025-01-01&sp=r&sv=2017-04-17&sr=b&sig=zy8L4shZa3XXBe152hPnhXsyfBqCufDOz01a9ZHWU28%3D"\
        --quiet

azcopy --source https://datasharesa.blob.core.windows.net/imagenet/train.csv \
        --destination  /data/imagenet/train.csv\
        --source-sas "?se=2025-01-01&sp=r&sv=2017-04-17&sr=b&sig=EUcahDDZcefOKtHoVWDh7voAC1BoxYNM512spFmjmDU%3D"\
        --quiet

azcopy --source https://datasharesa.blob.core.windows.net/imagenet/train.tar.gz \
        --destination  /data/imagenet/train.tar.gz\
        --source-sas "?se=2025-01-01&sp=r&sv=2017-04-17&sr=b&sig=qP%2B7lQuFKHo5UhQKpHcKt6p5fHT21lPaLz1O/vv4FNU%3D"\
        --quiet

cd /data/imagenet
tar -xzf train.tar.gz
tar -xzf validation.tar.gz