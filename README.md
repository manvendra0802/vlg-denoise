# Image Denoising using Residual Block Architecture

## Training Models
The training code is in file'training.py'.
Here, first we trained the model using the radial crop technique, resulting in the `denoising1model.h5` file. Further, we applied a fine-tuning technique to train the model 'denoising_fine_tuned_model.h5' file , which is our final trained model.

## Testing Part
Our model has a `main.py` file which takes noisy images as input from the `/test/low` directory and predicts the corresponding denoised images, saving them to the `/test/predicted` directory.

## Report
The attached report clearly explains every part of the code and its functionality.

## Conclusion
Our major criterion in this project was the PSNR score, which was evaluated as the benchmark. For our model, we found a (maximum) psnr=23.85 and val_psnr=23.41 while training the'denoising1model.h5' which reduced a bit during fine tuning and training the fine_tuned model .
