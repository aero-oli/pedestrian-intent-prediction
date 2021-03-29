#!/bin/bash

####################################################################################
######################### UPDATE PATHS BELOW BEFORE RUNNING ########################
####################################################################################

# Provide full path to PedestrianActionBenchmark repository
BENCHMARK_REPO_PATH=/home/sourab/Data/repos/PedestrianActionBenchmark/ # e.g. /home/user/PedestrainActionBenchmark/

####################################################################################
########################### DO NOT MODIFY SETTINGS BELOW ###########################
####################################################################################

# Step 0: Setup Environment
touch benchmark_log.txt
echo "Automating benchmarking of existing methods from the \"Pedestrian Action Benchmark\" paper by training and testing each neural network atleast five times" |& tee -a benchmark_log.txt
echo "Full Path to the Pedestrian Action Benchmark repository is set as follows: ${BENCHMARK_REPO_PATH}" |& tee -a benchmark_log.txt
echo "Changing Current Directory to ${BENCHMARK_REPO_PATH}" |& tee -a benchmark_log.txt
cd ${BENCHMARK_REPO_PATH} |& tee -a benchmark_log.txt
git status |& tee -a benchmark_log.txt
rm -rf ./models/* |& tee -a benchmark_log.txt

# Step 1: Download and extract PIE and JAAD datasets
echo "Step 1: Download and extract PIE and JAAD datasets" |& tee -a benchmark_log.txt
echo "Current Assumption: This step is already done!" |& tee -a benchmark_log.txt

# Step 2: Download Python data interface
echo "Step 2: Download Python data interface" |& tee -a benchmark_log.txt
echo "Current Assumption: This step is already done!" |& tee -a benchmark_log.txt

# Step 3: Install Docker
echo "Step 3: Install Docker" |& tee -a benchmark_log.txt
echo "Current Assumption: This step is already done!" |& tee -a benchmark_log.txt

# Step 4: Change Permission for scripts in the docker folder
echo "Step 4: Change Permission for scripts in the docker folder" |& tee -a benchmark_log.txt
chmod +x docker/*.sh |& tee -a benchmark_log.txt

# Step 5: Build Docker Image
echo "Step 5: Build Docker Image" |& tee -a benchmark_log.txt
docker/build_docker.sh |& tee -a benchmark_log.txt

# Step 6: Set Paths for PIE and JAAD datasets in docker/run_docker.sh
echo "Step 6: Set Paths for PIE and JAAD datasets in docker/run_docker.sh" |& tee -a benchmark_log.txt
echo "Current Assumption: This step is already done!" |& tee -a benchmark_log.txt

# Step 7: Run Docker Container
echo "Step 7: Run Docker Container" |& tee -a benchmark_log.txt
docker/run_docker -sh 0 |& tee -a benchmark_log.txt

# Step 8: Exit Docker Container
echo "Step 8: Exit Docker Container" |& tee -a benchmark_log.txt
exit |& tee -a benchmark_log.txt
