#!/bin/bash

##############################################################################################################
##################################### UPDATE BELOW PATHS BEFORE RUNNING ######################################
##############################################################################################################

# Provide full path to the darknet repository
DARKNET_REPO_PATH=/home/sourab/Data/temp3/darknet                           #e.g. /home/user/darknet

# Provide full paths to the road-object-detection-using-yolo-v4 repository
ROAD_REPO_PATH=/home/sourab/Data/temp3/road-object-detection-using-yolov4   #e.g. /home/user/repo-name

# Provide full path to the dataset
DATA_PATH=/home/sourab/Data/dataset/DeepDrive                               #e.g. /home/user/DeepDrive

# Provide full path to the script folder
SCRIPT_FOLDER=/home/sourab/Data/repos/master-thesis/scripts            #e.g. /home/user/master-thesis/scripts

##############################################################################################################
##################################### DO NOT MODIFY THE SETTINGS BELOW #######################################
##############################################################################################################

# Cleanup environment
echo "Cleaning up environment..."
echo ""
rm -rf $DARKNET_REPO_PATH
rm -rf $ROAD_REPO_PATH

# Print current path and provided path
echo "Path of the bash script: $SCRIPT_FOLDER"
echo "Path to the darknet repository: $DARKNET_REPO_PATH"
echo "Path to the road-object-detection-using-yolov4 repository: $ROAD_REPO_PATH"
echo ""

# Clone the repositories
echo "Cloning the repositories..."
echo ""
git clone https://github.com/AlexeyAB/darknet.git $DARKNET_REPO_PATH
git clone https://github.com/sourabbapusridhar/road-object-detection-using-yolov4.git $ROAD_REPO_PATH

# Update Makefile based on requirements
echo "Updating makefile based on requirements..."
echo ""
cd $DARKNET_REPO_PATH
sed -i 's/OPENCV=0/OPENCV=1/' Makefile && echo "Flag to build with OpenCV updated"
sed -i 's/GPU=0/GPU=1/' Makefile && echo "Flag to build with GPU updated"
sed -i 's/CUDNN=0/CUDNN=1/' Makefile && echo "Flag to build with cuDNN updated"

# Build darknet
echo "Building darknet..."
echo ""
make && echo "Darknet build successful"

# Download pre-trained weights file
echo "Downloading pre-trained weights file..."
echo ""
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29 && echo "Pre-trained weights downloaded successfully!"

# Setup the Berkley DeepDrive dataset
echo "Setting up the Berkley DeepDrive dataset..."
echo ""
cp -rvi $DATA_PATH/bdd100k_images.zip $DARKNET_REPO_PATH/data/ && echo "Copied image data successfully!"
cp -rvi $DATA_PATH/bdd100k_labels_release.zip $DARKNET_REPO_PATH/data/ && echo "Copied label data successfully!"

# Unzip the dataset and the annotations in the data folder
echo "Unziping the dataset and the annotations in the data folder..."
echo ""
unzip $DARKNET_REPO_PATH/data/bdd100k_images.zip -d $DARKNET_REPO_PATH/data/ && echo "Images unzipped successfully!"
unzip $DARKNET_REPO_PATH/data/bdd100k_labels_release.zip -d $DARKNET_REPO_PATH/data/ && echo "Labels unzipped successfully!"

# Delete unwanted files
echo "Deleting unwanted files..."
echo ""
rm -rf $DARKNET_REPO_PATH/data/bdd100k_images.zip && echo "bdd100k_images.zip delected successfully!"
rm -rf $DARKNET_REPO_PATH/data/bdd100k_labels_release.zip && echo "bdd100k_labels-release.zip delected successfully!"
rm -rf $DARKNET_REPO_PATH/data/bdd100k/images/10k && echo "10k image folder deleted successfully!"

# Copy bdd100k.names files
echo "Copying bdd100k.names files..."
echo ""
cp -vi $ROAD_REPO_PATH/data/* $DARKNET_REPO_PATH/data/bdd100k/ && echo "Copied bdd100k.names successfully!"

# Convert labels from JSON files to text files
echo "Converting JSON files to text files..."
echo ""
python $ROAD_REPO_PATH/utils/convert_labels.py -ij $DARKNET_REPO_PATH/data/bdd100k/labels/bdd100k_labels_images_train.json -in $DARKNET_REPO_PATH/data/bdd100k/bdd100k.names -o $DARKNET_REPO_PATH/data/bdd100k/images/100k/train/ && echo "Training labels converted successfully!"
python $ROAD_REPO_PATH/utils/convert_labels.py -ij $DARKNET_REPO_PATH/data/bdd100k/labels/bdd100k_labels_images_val.json -in $DARKNET_REPO_PATH/data/bdd100k/bdd100k.names -o $DARKNET_REPO_PATH/data/bdd100k/images/100k/val/ && echo "Validation labels converted successfully!"

# Remove data without annotations
echo "Removing data without annotations..."
echo ""
python $ROAD_REPO_PATH/utils/data_cleanup.py -i $DARKNET_REPO_PATH/data/bdd100k/images/100k/train/ && echo "Removed unwanted training data successfully!"
python $ROAD_REPO_PATH/utils/data_cleanup.py -i $DARKNET_REPO_PATH/data/bdd100k/images/100k/val/ && echo "Removed unwanted validation data successfully!"

# Generate paths for training and validation images
echo "Generating paths for training and validation images..."
echo ""
python $ROAD_REPO_PATH/utils/generate_paths.py -it data/bdd100k/images/100k/train/ -iv data/bdd100k/images/100k/val/ -o $DARKNET_REPO_PATH/data/bdd100k/ && echo "Generated paths for training and validation images successfully!"

# Generate data file containing relative paths to the training, validation and backup folders for YOLOv4
echo "Generating data file containing relative paths to the training, validation and backup folders for YOLOv4..."
echo ""
python $ROAD_REPO_PATH/utils/generate_data_file.py -c 10 -t data/bdd100k/bdd100k_train.txt -v data/bdd100k/bdd100k_val.txt -n data/bdd100k/bdd100k.names -b backup/ -o $DARKNET_REPO_PATH/data/bdd100k/

# Copy pre-defined YOLOv4 network configuration file to cfg folder
echo "Copying pre-defined YOLOv4 network configuration file to cfg folder..."
echo ""
cp -vi $ROAD_REPO_PATH/config/* $DARKNET_REPO_PATH/cfg/ && echo "Pre-defined YOLOv4 config copied successfully!"

# Train YOLOV4 on Berkley DeepDrive dataset
echo "Training YOLOv4 on Berkley DeepDrive dataset..."
echo ""
chmod +x darknet
./darknet detector train data/bdd100k/bdd100k.data cfg/yolov4-tiny-bdd100k.cfg yolov4-tiny.conv.29 -dont_show -map

# Exit script
echo "Exiting script..."
echo ""
return 0
