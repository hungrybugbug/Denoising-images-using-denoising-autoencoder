--------------------------------------------------
Denoising Autoencoder on CIFAR-10
--------------------------------------------------

Author: Mohammad Sooban
Course: [Enter Course Name]
Instructor: [Enter Instructor Name]
Institution: [Enter University Name]
Semester: [Enter Semester]

--------------------------------------------------
Project Description
--------------------------------------------------

This project implements a Convolutional Denoising Autoencoder (DAE) to 
remove noise from images in the CIFAR-10 dataset. The model is trained to 
reconstruct clean images from noisy inputs corrupted by Gaussian and 
salt-and-pepper noise.

The system uses deep learning techniques based on PyTorch and follows a 
research-oriented experimental workflow.

--------------------------------------------------
Objectives
--------------------------------------------------

1. Load and preprocess the CIFAR-10 dataset.
2. Apply artificial noise to training images.
3. Design and train a denoising autoencoder.
4. Evaluate reconstruction quality using quantitative metrics.
5. Analyze the effect of noise level and bottleneck size.
6. Present results in a technical report.

--------------------------------------------------
Project Files
--------------------------------------------------

1. denoising_autoencoder.py / notebook.ipynb
   - Main implementation and experiments.

2. report.tex
   - LNCS LaTeX report file.

3. prompts.txt
   - List of AI prompts used during the project.

4. README.txt
   - Project documentation (this file).

5. reconstruction_examples.png
   - Visualization of denoising results.

--------------------------------------------------
Requirements
--------------------------------------------------

Python 3.8+
PyTorch
Torchvision
NumPy
Matplotlib
Scikit-image
Pandas

(Optional) GPU with CUDA support for faster training.

--------------------------------------------------
How to Run the Project
--------------------------------------------------

1. Install required libraries:

   pip install torch torchvision numpy matplotlib scikit-image pandas

2. Run the main script or notebook:

   python denoising_autoencoder.py

   OR

   Open notebook.ipynb in Jupyter/Colab and run all cells.

3. The script will:
   - Download CIFAR-10 automatically
   - Train the model
   - Generate plots
   - Display reconstruction results
   - Produce experimental tables

--------------------------------------------------
Experimental Settings
--------------------------------------------------

Batch Size: 128
Optimizer: Adam
Learning Rate: 0.001
Loss Function: Mean Squared Error (MSE)
Epochs: 20–40
Latent Dimensions: 64, 128, 256
Noise Levels: 0.05, 0.1, 0.2

--------------------------------------------------
Evaluation Metrics
--------------------------------------------------

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

--------------------------------------------------
Results Summary
--------------------------------------------------

The model successfully removes a significant amount of noise from CIFAR-10 
images. Increasing the bottleneck size improves reconstruction quality, 
while higher noise levels reduce performance.

Outputs are slightly blurred due to MSE loss and compression effects.

--------------------------------------------------
Limitations
--------------------------------------------------

- Blurry reconstructions
- Limited network depth
- No skip connections
- No perceptual loss

--------------------------------------------------
Future Improvements
--------------------------------------------------

- Use deeper CNN architectures
- Add U-Net skip connections
- Use perceptual loss
- Explore GAN-based denoising
- Increase training epochs

--------------------------------------------------
Acknowledgment
--------------------------------------------------

This project was completed as part of an academic assignment using 
AI-assisted coding and guidance.

--------------------------------------------------
End of File
--------------------------------------------------
