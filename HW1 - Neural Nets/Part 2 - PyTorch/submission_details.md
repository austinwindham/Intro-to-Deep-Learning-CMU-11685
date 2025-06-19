# HW 1 Part 2 Submission Details
In this homework, i created a basic multilayer perceptron neural network structure to predict phonemes from spectrograms of audio files. The different methods I used for creating the network are below. All of my code was done in the python notebook template file, so to run my code, you just need to run the code blocks in sequential order. 

## Methods
### Architecture
We do not really have many different architectures to choose from in this assignment. At first, I created  cylinder architectures and tested them, but I was not quite getting to 86 % validation, so I tried with a diamond and a pyramid architecture.  These seemed to perform worse than the cylinder architecture, so I just increased my cylinder architecture till I was slightly below the maximum amount of parameters. 

### Hyperparameters
I would typically run for around 10 epochs and stop the code if it was not performing better than other tests. I also tested with different learning schedulers. Most of my team was using the basic scheduler, so I tried cosine annealing with restarts and this seemed like it was doing slightly better. I extended the cosine period parameter to 8 and the multiplier to 2, so two full periods was 24 epochs. This was giving me my best results. I felt like the cosine annealing was decresing the learning rate at the best rate and then the restart allowed the network a second chance to make slight improvements. I was not getting much of a difference between relu and gelu, so I just went with relu. Maybe, I should have tried signmoid. I never really varied batch size or initial learning rate. I chose pretty much just standard values of 1024 and 0.001. I adjusted dropout some. I wish I had more time to test with this. I would have liked to train the network where the dropoiut number changes between epochs. I adjusted the context above and below my initial value of 30, and I was not seeing any benefit, so I just kept it at 30. 

### Data Handling
I did not do anything special with data augmentations beyond hte initial stuff. I varied frequency and time masking some, but was not seeing much of an improvement and neither did my teammates. 

### Best Run
Ultimately, my best run gave me slightly above 86 %. I used cosine annealing with restarts with a T_0 of 8 and T_mul of 2. I trained for 24 epochs with batch size 1024 with a cylinder architecture. My optimizer was AdamW, my learning rate was 0.001, and my dropout was .001. I also used only Relu activation function. 

## Links
### Weights & Biases (WandB) Project
The link to my weights and biases for my expirments is here:
[View my WandB project here](https://wandb.ai/awindham4-carnegie-mellon-university/hw1p2?nw=nwuserawindham4)

### Other Files
An excel sheet of some of our ablations for my team is included in this tar file along with my python notebook. 



