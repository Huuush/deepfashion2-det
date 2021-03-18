## Introduction
Use mmdet to detect cloth

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation. Or, if you use docker container, see this [dockerfile](docker/Dockerfile)

## Getting Started

1.First install mmdetection follow the above installation step.
2.please download deepfasion2 dataset [here](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok) and annotation file [here]()
3.make the file like this.
```plain
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── deepfashion2_zip
│   │   ├── annotations
│   │   ├── train
│   │   ├── validation
│   │   ├── test
```
Note that train dir is a dir include all images. and validation dir could also do like this:
(1)Make a dir named 'train' in data/deepfashion2_zip.(maybe you need to rename the dir name for deepfashion2_zip when you download the dataset)
(2)Unzip the train.zip in deepfashion2 dataset, there is a annos dir and a image dir.
(3)ln -s /image dir/ /data/deepfashion2_zip/train/

5.run the train command followed by [here](docs/1_exist_data_model.md): Training on multiple GPUs.(a machine with n GPUs)
```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
    
for me:
bash ./tools/dist_train.sh configs/vfnet/vfnet_r2_101_fpn_2x_cloth.py 8
```

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.
