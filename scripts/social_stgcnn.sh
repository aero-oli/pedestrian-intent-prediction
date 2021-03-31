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

# Print provided path
echo "Path to the Social-STGCNN repository: $SOCIAL_STGCNN_REPO"
echo ""

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

# Benchmark Model based on the pretrained weights
echo "Benchmarking models based on pretrained weights..."
echo ""
cd  $SOCIAL_STGCNN_REPO
python test.py |& tee test_pretrained_weights.txt

# Train model for each dataset with the best configuration from the paper
echo "Training model for each dataset with the with the best configuration from the paper"
echo ""

for loop_counter in $(seq 1 $SEQ_LEN)
do

echo "---------------------------- TRAINING NUMBER: $loop_counter ----------------------------"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset eth --tag social-stgcnn-eth --use_lrschd --num_epochs 250 && echo "eth Launched." |& tee training_eth_${loop_counter}.txt &
P0=$! 

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset hotel --tag social-stgcnn-hotel --use_lrschd --num_epochs 250 && echo "hotel Launched." |& tee training_hotel_${loop_counter}.txt &
P1=$! |& tee training_

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset univ --tag social-stgcnn-univ --use_lrschd --num_epochs 250 && echo "univ Launched." |& tee training_univ_${loop_counter}.txt &
P2=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara1 --tag social-stgcnn-zara1 --use_lrschd --num_epochs 250 && echo "zara1 Launched." |& tee training_zara1_${loop_counter}.txt &
P3=$!

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset zara2 --tag social-stgcnn-zara2 --use_lrschd --num_epochs 250 && echo "zara2 Launched." |& tee training_zara2_${loop_counter}.txt &
P4=$!

wait $P0 $P1 $P2 $P3 $P4

python test.py |& test_${loop_counter}.txt

done

# Deactivate Conda Environment
echo "Deactivating Conda Environment..."
echo ""
conda deactivate

