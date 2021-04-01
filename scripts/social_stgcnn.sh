#!/bin/bash

####################################################################################################
################################### UPDATE PATHS BELOW BEFORE RUNNING ##############################
####################################################################################################

# Provide full path to the Social-STGCNN repository
SOCIAL_STGCNN_REPO=/home/sourab/Data/test_folder/Social-STGCNN #e.g. /home/user/repos/SOCIAL-STGCNNN
SOCIAL_STGCNN_VENV=Social-STGCNN                               #e.g. Social-STGCNN
SEQ_LEN=1                                                      #e.g. 10,20

####################################################################################################
################################### DO NOT MODIFY SETTINGS BELOW ###################################
####################################################################################################

# Cleanup (Temproary Step)
echo "Cleaning up environment (Temproary Step)"
echo ""
rm -rf ~/Data/test_folder/*
conda remove --name $SOCIAL_STGCNN_VENV --all --yes

# Print current path and provided path
echo "Path of the bash script: $PWD"
echo "Path to the Social-STGCNN repository: $SOCIAL_STGCNN_REPO"
echo ""

# Cleanup Output Folder
echo "Cleaning up output folder..."
echo ""
OUTPUT_PATH="$(dirname "PWD")/benchmark/$SOCIAL_STGCNN_VENV"
rm -rf $OUTPUT_PATH
mkdir $OUTPUT_PATH

# Clone the repository
echo "Cloning the repository..."
echo ""
git clone https://github.com/abduallahmohamed/Social-STGCNN.git $SOCIAL_STGCNN_REPO

# Activate Conda Virtual Environment to run from Bash
echo "Activating Conda Virtual Environment to run from Bash"
echo ""
conda init bash

# Create a Conda Virtual Environment
echo "Creating a Conda Vritual Environment..."
echo ""
conda create --yes --name $SOCIAL_STGCNN_VENV python=3.6
conda activate $SOCIAL_STGCNN_VENV

# Install all dependecies
echo "Installing all dependecies..."
echo ""
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge --yes
pip install networkx
pip install numpy
pip install tqdm

# Benchmark Model based on the pretrained weights and store initial results
echo "Benchmarking models based on pretrained weights and storing initial results..."
echo ""
cd  $SOCIAL_STGCNN_REPO
python test.py |& tee test_pretrained_weights.txt
mv -v $SOCIAL_STGCNN_REPO/test_pretrained_weights.txt $SOCIAL_STGCNN_REPO/checkpoint/

# Train model for each dataset with the best configuration from the paper and store the results
echo "Training model for each dataset with the with the best configuration from the paper and storing the results..."
echo ""

for loop_counter in $(seq 1 $SEQ_LEN)
do

echo "---------------------------- TRAINING NUMBER: $loop_counter ----------------------------"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 250 |& tee training_eth_${loop_counter}.txt && echo "eth Launched." &
P0=$! 

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag social-stgcnn-hotel --use_lrschd --num_epochs 250 |& tee training_hotel_${loop_counter}.txt && echo "hotel Launched." &
P1=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag social-stgcnn-univ --use_lrschd --num_epochs 250 |& tee training_univ_${loop_counter}.txt && echo "univ Launched." &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag social-stgcnn-zara1 --use_lrschd --num_epochs 250 |& tee training_zara1_${loop_counter}.txt && echo "zara1 Launched." &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag social-stgcnn-zara2 --use_lrschd --num_epochs 250 |& tee training_zara2_${loop_counter}.txt && echo "zara2 Launched." &
P4=$!

wait $P0 $P1 $P2 $P3 $P4

python test.py |& tee test_${loop_counter}.txt

mv -v $SOCIAL_STGCNN_REPO/training_eth_${loop_counter}.txt $SOCIAL_STGCNN_REPO/checkpoint/social-stgcnn-eth/
mv -v $SOCIAL_STGCNN_REPO/training_hotel_${loop_counter}.txt  $SOCIAL_STGCNN_REPO/checkpoint/social-stgcnn-hotel/
mv -v $SOCIAL_STGCNN_REPO/training_univ_${loop_counter}.txt  $SOCIAL_STGCNN_REPO/checkpoint/social-stgcnn-univ/
mv -v $SOCIAL_STGCNN_REPO/training_zara1_${loop_counter}.txt  $SOCIAL_STGCNN_REPO/checkpoint/social-stgcnn-zara1/
mv -v $SOCIAL_STGCNN_REPO/training_zara2_${loop_counter}.txt  $SOCIAL_STGCNN_REPO/checkpoint/social-stgcnn-zara2/
cp -rv $SOCIAL_STGCNN_REPO/checkpoint/* $OUTPUT_PATH

done

# Deactivate Conda Environment
echo "Deactivating Conda Environment..."
echo ""
conda deactivate

