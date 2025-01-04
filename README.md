# ASRMF:Adaptive image super-resolution based on dynamic-parameter DNN with multi-feature prior Signal Processing  
## Usage
### Dataset preparation  
Modify the Settings in `options/train_mysrnet.json`  
Set the paths to the local project root, LR, and HR, respectively
### Train
1. Attention! During the training phase, uncomment the four lines of code in 'models/modules/sft_arch.py', which begin with fea1, fea2, fea3, fea4. (Marked in the py file)  
2. Run the following command on the terminal  
`python train.py`
### Test
1. Attention! During the test phase, comment out the four lines of code in 'models/modules/sft_arch.py', starting with fea1, fea2, fea3, fea4. (Marked in the py file)
2. The test_img_folder variable in the test.py file requires the HR image path to be entered, after which the default quadruple downsampling is performed.
3. Run the following command on the terminal  
`python test.py`
