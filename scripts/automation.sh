#!/bin/bash

###########################################################################################################
###################################### UPDATE PATHS BEFORE RUNNING ########################################
###########################################################################################################

# Provide full path to the Pedestrian Action Benchmark script
PED_SCRIPT_PATH=/home/sourab/Data/repos/master-thesis/scripts/pedestrian_action_benchmark.sh

# Provide full path to the Social-STGCNN script
SOCIAL_SCRIPT_PATH=/home/sourab/Data/repos/master-thesis/scripts/social_stgcnn.sh

# Provide full path to the YOLOv4 training script
YOLO_SCRIPT_PATH=/home/sourab/Data/repos/master-thesis/scripts/yolov4.sh

###########################################################################################################
#################################### DO NOT MODIFY THE SETTINGS BELOW #####################################
###########################################################################################################

# Sanity check: Check if all the above mentioned path exists
if [[ -f $PED_SCRIPT_PATH ]] && [[ -f $SOCIAL_SCRIPT_PATH ]] && [[ -f $YOLO_SCRIPT_PATH ]]
then

# Run the Pedestrian Action Benchmark Script
echo "Running the Pedestrian Action Benchmark Script..."
echo ""
source $PED_SCRIPT_PATH

# Run the Social-STGCNN Script
echo "Running the Social-STGCNN Script..."
echo ""
source $SOCIAL_SCRIPT_PATH

# Run the YOLOv4 Training Script
echo "Running the YOLOv4 training script..."
echo ""
source $YOLO_SCRIPT_PATH

else

# Error: Path does not exists!
echo "Path does not exists!"
echo ""

fi

# Exit the Script
echo "Exiting the script..."
echo ""
exit 0

