# Master Thesis
===============

[![Project Status: Active – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

Repository for master thesis project titled "Pedestrian Intent Prediction Using Deep Machine Learning" at the Chalmers University of Technology.

## Abstract 
*To be added*

## Datasets
### Joint Attention in Autonomous Driving (JAAD) Dataset [[Link](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/)]:
JAAD is a dataset for studying joint attention in the context of autonomous driving. The focus is on pedestrian and driver behaviors at the point of crossing and factors that influence them. To this end, JAAD dataset provides a richly annotated collection of 346 short video clips (5-10 sec long) extracted from over 240 hours of driving footage. These videos filmed in several locations in North America and Eastern Europe represent scenes typical for everyday urban driving in various weather conditions. Bounding boxes with occlusion tags are provided for all pedestrians making this dataset suitable for pedestrian detection.

Behavior annotations specify behaviors for pedestrians that interact with or require attention of the driver. For each video there are several tags (weather, locations, etc.) and timestamped behavior labels from a fixed list (e.g. stopped, walking, looking, etc.). In addition, a list of demographic attributes is provided for each pedestrian (e.g. age, gender, direction of motion, etc.) as well as a list of visible traffic scene elements (e.g. stop sign, traffic signal, etc.) for each frame. 

- **Full Dataset:** http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
- **Annotations:** https://github.com/ykotseruba/JAAD

## Requirements
The code is based on Python3 (>=3.8). There are a few dependencies to run the code. The major libraries are listed as follows:
* Tensorflow (>=2.3.0)

## Installation Guide
To install the anaconda environment, navigate to the repository folder, and run the following command in the terminal:

```
$conda env create -f environment.yml
```

## Execution Guide
1. To activate the Conda environment, please run the following command in the terminal:

```
$conda activate intent
```

2. Train or Test the network in the Conda environment using the below command in terminal:

```
$python train.py -h     # Command to understand command line arguments
$python test.py -h      # Command to understand command line arguments
```

3. To deactivate the Conda environment, please run the following command in the terminal:

```
$conda deactivate
```

## Clean-up Guide
To remove the anaconda environment, navigate to the repository folder, and run the following command in the terminal:

```
$conda remove --name intent --all
```

## Authors
* Aren Moosakhanian
* Sourab Bapu Sridhar

## License
This project is released under the terms of [MIT License](LICENSE).