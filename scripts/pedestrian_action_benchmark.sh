#!/bin/bash

#########################################################################################################
################################### UPDATE PATHS BELOW BEFORE RUNNING ###################################
#########################################################################################################

# Provide full path to PedestrianActionBenchmark repository
BENCHMARK_REPO_PATH=/home/sourab/Data/test_folder2/PedestrianActionBenchmark #e.g. /home/user/PedestrainActionBenchmark
JAAD_REPO_PATH=/home/Datasets/MLDatasetsStorage/JAAD                        #e.g. /home/user/JAAD
PIE_REPO_PATH=/home/Datasets/MLDatasetsStorage/PIE                          #e.g. /home/user/PIE
BENCHMARK_VENV=Pedestrian_Action_Benchmark                                  #e.g. Pedestrian_Action_Benchmark
SEQ_LEN=1                                                                   #e.g. 10,20,...
 
#########################################################################################################
##################################### DO NOT MODIFY SETTINGS BELOW ######################################
#########################################################################################################

# Cleanup (Temproary Step)
echo "Cleaning up environment (Temproary Step)"
echo ""
rm -rf ~/Data/test_folder2/*
docker system prune -af

# Print current path and provided path
echo "Path of the bash script: $PWD"
echo "Path to the Pedestrian Action Benchmark repository: $BENCHMARK_REPO_PATH"
echo "Path to the JAAD repository: $JAAD_REPO_PATH"
echo "Path to the PIE repository: $PIE_REPO_PATH"
echo ""


# Cleanup output folder
echo "Cleaning up output folder..."
echo ""
OUTPUT_PATH="$(dirname "$PWD")/benchmark/$BENCHMARK_VENV"
rm -rf $OUTPUT_PATH
mkdir $OUTPUT_PATH

# Clone the repository
echo "Cloning the repository..."
echo ""
git clone https://github.com/ykotseruba/PedestrianActionBenchmark.git $BENCHMARK_REPO_PATH

# Download python data interfaces
echo "Downloading python data interfaces..."
echo ""
cp -v $JAAD_REPO_PATH/jaad_data.py $BENCHMARK_REPO_PATH/
cp -v $PIE_REPO_PATH/pie_data.py $BENCHMARK_REPO_PATH/

# Change Permission for scripts in the docker folder
echo "Changing permission for scripts in the docker folder..."
echo ""
cd $BENCHMARK_REPO_PATH
chmod +x docker/*.sh

# Build Docker Image
echo "Building Docker Images..."
echo ""
source docker/build_docker.sh

# Set paths for PIE and JAAD Datasets in docker/run_docker.sh
echo "Setting paths for PIE and JAAD Datasets in docker/run_docker.sh..."
echo ""
sed -i "s@PIE_DATA=.*@PIE_DATA=$PIE_REPO_PATH@" $BENCHMARK_REPO_PATH/docker/run_docker.sh && echo "PIE_DATA successfully set!"
sed -i "s@JAAD_DATA=.*@JAAD_DATA=$JAAD_REPO_PATH@" $BENCHMARK_REPO_PATH/docker/run_docker.sh && echo "JAAD_DATA successfully set!"
sed -i "s@MODELS=.*@MODELS=$OUTPUT_PATH@" $BENCHMARK_REPO_PATH/docker/run_docker.sh && echo "MODELS successfully set!"
sed -i "s@CODE_FOLDER=.*@CODE_FOLDER=$BENCHMARK_REPO_PATH@" $BENCHMARK_REPO_PATH/docker/run_docker.sh && echo "CODE_FOLDER successfully set!"
sed -i "s/-it/-d" $BENCHMARK_REPO_PATH/docker/run-docker.sh && echo "Docker Mode Successfully changed from iteractive to detached!"
echo ""

# Update configuration files for training and testing
echo "Updating configuration files for training and testing..."
echo ""
echo "Updating configuration for ATGC model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/ATGC.yaml && echo "ATGC dataset updated"
sed -i "s/\[32,/\[/" $BENCHMARK_REPO_PATH/config_files/ATGC.yaml && echo "ATGC batch_size updated"
sed -i "s/\[20,/\[/" $BENCHMARK_REPO_PATH/config_files/ATGC.yaml && echo "ATGC epochs updated"
sed -i "s/\[0.000005,/\[/" $BENCHMARK_REPO_PATH/config_files/ATGC.yaml && echo "ATGC lr updated"
echo ""

echo "Updating configuration for C3D model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/C3D.yaml && echo "C3D dataset updated"
sed -i "s/\[16,/\[/" $BENCHMARK_REPO_PATH/config_files/C3D.yaml && echo "C3D batch_size updated"
sed -i "s/\[40,/\[/" $BENCHMARK_REPO_PATH/config_files/C3D.yaml && echo "C3D epochs updated"
sed -i "s/\[5.0e-05,/\[/" $BENCHMARK_REPO_PATH/config_files/C3D.yaml && echo "C3D lr updated"
echo ""

echo "Updating configuration for ConvLSTM_vgg16 model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/ConvLSTM_vgg16.yaml && echo "ConvLSTM dataset updated"
sed -i "s/\[2,/\[/" $BENCHMARK_REPO_PATH/config_files/ConvLSTM_vgg16.yaml && echo "ConvLSTM batch_size updated"
sed -i "s/\[10,/\[/" $BENCHMARK_REPO_PATH/config_files/ConvLSTM_vgg16.yaml && echo "ConvLSTM epochs updated"
sed -i "s/\[0.00005,/\[/" $BENCHMARK_REPO_PATH/config_files/ConvLSTM_vgg16.yaml && echo "ConvLSTM lr updated"
echo ""

echo "Updating configuration for PCPA model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/PCPA.yaml && echo "PCPA dataset updated"
sed -i "s/\[8,/\[/" $BENCHMARK_REPO_PATH/config_files/PCPA.yaml && echo "PCPA batch_size updated"
sed -i "s/\[80,/\[/" $BENCHMARK_REPO_PATH/config_files/PCPA.yaml && echo "PCPA epochs updated"
sed -i "s/\[1.0e-06,/\[/" $BENCHMARK_REPO_PATH/config_files/PCPA.yaml && echo "PCPA lr updated"
echo ""

echo "Updating configuration for SFRNN model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/SFRNN.yaml && echo "SFRNN dataset updated"
sed -i "s/\[32,/\[/" $BENCHMARK_REPO_PATH/config_files/SFRNN.yaml && echo "SFRNN batch_size updated"
sed -i "s/\[60,/\[/" $BENCHMARK_REPO_PATH/config_files/SFRNN.yaml && echo "SFRNN epochs updated"
sed -i "s/\[0.00005,/\[/" $BENCHMARK_REPO_PATH/config_files/SFRNN.yaml && echo "SFRNN lr updated"
echo ""

echo "Updating configuration for Static_vgg16 model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/Static_vgg16.yaml && echo "Static dataset updated"
sed -i "s/\[32,/\[/" $BENCHMARK_REPO_PATH/config_files/Static_vgg16.yaml && echo "Static batch_size updated"
sed -i "s/\[20,/\[/" $BENCHMARK_REPO_PATH/config_files/Static_vgg16.yaml && echo "Static epochs updated"
sed -i "s/\[0.000005,/\[/" $BENCHMARK_REPO_PATH/config_files/Static_vgg16.yaml && echo "Static lr updated"
echo ""

echo "Updating configuration for Two_Stream model..."
sed -i "s/\[pie,/\[/" $BENCHMARK_REPO_PATH/config_files/Two_Stream.yaml && echo "TwoStream dataset updated"
sed -i "s/\[16,/\[/" $BENCHMARK_REPO_PATH/config_files/Two_Stream.yaml && echo "TwoStream dataset updated"
sed -i "s/\[60,/\[/" $BENCHMARK_REPO_PATH/config_files/Two_Stream.yaml && echo "TwoStream dataset updated"
sed -i "s/\[0.0000005,/\[/" $BENCHMARK_REPO_PATH/config_files/Two_Stream.yaml && echo "TwoStream dataset updated"
echo ""

echo "Updating configuration for configs_default model..."
sed -i "s/dataset:.*/dataset:\sjaad/" $BENCHMARK_REPO_PATH/config_files/configs_default.yaml && echo "Default dataset updated"
echo ""

# Run docker/run_docker.sh
echo "Running docker/run_docker.sh..."
echo ""
source $BENCHMARK_REPO_PATH/docker/run_docker.sh -gd 0

# Run the shell script from the docker container
echo "Running the shell script from the docker container..."
echo ""

# Train and test models
echo "Training and testing models..."
echo ""

# Store metrics
echo "Storing metrics..."
echo ""

# Exit Docker Container
echo "Exiting Docker container..."
echo ""
