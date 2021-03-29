#!/bin/bash

#######################################################################################
############################## UPDATE PATHS BEFORE RUNNING ############################
#######################################################################################

# Provide absolute paths of all the scripts automating individual tasks

BENCHMARK_PATH=/home/sourab/Data/repos/master-thesis/utils          # e.g. /home/user/PedestrianActionBenchmark
YOLOV4_DEEPDRIVE_PATH=/home/sourab/Data/repos/master-thesis/utils   # e.g. /home/user/PedestrianActionBenchmark
YOLOV3_JAAD_PATH=/home/sourab/Data/repos/master-thesis/utils        # e.g. /home/user/PedestrianActionBenchmark
YOLOV4_JAAD_PATH=/home/sourab/Data/repos/master-thesis/utils        # e.g. /home/user/PedestrianActionBenchmark


#######################################################################################
############################ DO NOT MODIFY SETTINGS BELOW #############################
#######################################################################################

# Task 0: Create a log file
touch log.txt
echo "The absolute paths are set as follows" >> log.txt
echo "BENCHMARK_PATH: ${BENCHMARK_PATH}" >> log.txt
echo "YOLOV4_DEEPDRIVE_PATH: ${YOLOV4_DEEPDRIVE_PATH}" >> log.txt
echo "YOLOV3_JAAD_PATH: ${YOLOV3_JAAD_PATH}" >> log.txt
echo "YOLOV4_JAAD_PATH: ${YOLOV4_JAAD_PATH}" >> log.txt 

# Task 1: Benchmark all methods from "Pedestrian Action Prediction" paper atleast 5 times
# To be added

# Task 2: Train and Test YOLOv4 Network on Berkley DeepDrive Dataset
# To be added

# Task 3: Train and Test YOLOv3 Network on JAAD Dataset
# To be added

# Task 4: Train and Test YOLOv4 Network on JAAD Dataset
# To be added
