# DQN_RL
Deep Q Learning Network to play Breakout

## Environment Installation
For Open AI Gym Atari environment, 
`$ pip install opencv-python-headless gym==0.10.4 gym[atari]`
Reference - [OpenAI's page](https://github.com/openai/gym)

## Goal
Implement DQN to play Breakout using Pytorch.
`$ python main.py --train_dqn`
`$ python test.py --test_dqn`

## Implementation
Input - Stack of 4 frames of gameplay snapshot
Number of actions - 4
System specs - Pytorch 1.11 with CUDA toolkit 11.3 on NVIDIA GeForce RTX 3060 GPU

### Model - Deep Q Learning Network
3 convolutional layers + 3 Linear fully connected layers. ReLU after each layer. Tensor is flattened after 3 convolutional layers to make it comoplatible with the linear layers.

Layer 1 - conv1
Input channels = 4 , Output channels = 32 , Kernel size = 8 , Stride = 4
Layer 2 – conv2
Input channels = 32 , Output channels = 64 , Kernel size = 4 , Stride = 2
Layer 3 – conv3
Input channels = 64 , Output channels = 64 , Kernel size = 3 , Stride = 1
Layer 4 – fc1
Input channels = 3136 , Output channels = 512 
Layer 5 – fc2
Input channels = 512 , Output channels = 128
Layer 6 – fc3
Input channels = 128 , Output channels = 4

## Results
Buffer size = 50000
Epsilon decay = 1e(-8)

### Training
Training Time for final model ==> 20 hours 15 mins
Number of episodes ==> About 70k (69,470)
Average Reward over 100 episodes ( training ) ==> 15


### Testing
Average Reward over 100 episodes ==> 49.28
Maximum Reward obtained in an episode ==> 256.0