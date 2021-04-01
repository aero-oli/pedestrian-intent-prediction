#!/bin/bash

#########################################################################################################
################################### UPDATE PATHS BELOW BEFORE RUNNING ###################################
#########################################################################################################

# Provide full path to PedestrianActionBenchmark repository
BENCHMARK_REPO_PATH=/home/sourab/Data/test_folder/PedestrianActionBenchmark #e.g. /home/user/PedestrainActionBenchmark
JAAD_REPO_PATH=/home/Datasets/MLDatasetsStorage/JAAD                        #e.g. /home/user/JAAD
PIE_REPO_PATH=/home/Datasets/MLDatasetsStorage/PIE                          #e.g. /home/user/PIE
BENCHMARK_VENV=Pedestrian_Action_Benchmark                                  #e.g. Pedestrian_Action_Benchmark
SEQ_LEN=1                                                                   #e.g. 10,20,...
 
#########################################################################################################
##################################### DO NOT MODIFY SETTINGS BELOW ######################################
#########################################################################################################

# Cleanup (Temproary Step)

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
bash docker/build_docker.sh

# Set paths for PIE and JAAD Datasets in docker/run_docker.sh
echo "Setting paths for PIE and JAAD Datasets in docker/run_docker.sh..."
echo ""

# Update configuration files for training and testing
echo "Updating configuration files for training and testing..."
echo ""

# Run docker/run_docker.sh
echo "Running docker/run_docker.sh..."
echo ""

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

