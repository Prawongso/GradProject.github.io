# Facial Reconstruction and Mask Removal for Identity and Emotion Recognition

Welcome to the GitHub repository for our project on **Facial Reconstruction and Mask Removal for Identity and Emotion Recognition**! This project addresses the challenges posed by face masks in facial recognition and emotion detection systems, which have become increasingly relevant due to the widespread use of masks in public spaces, especially during global health crises like the COVID-19 pandemic.

## Project Overview

Our system integrates four key technologies to provide a comprehensive solution for facial analysis in masked scenarios:

1. **Face Mask Detection**: Using a lightweight MobileNetV2 model, the system detects whether a person is wearing a mask with high accuracy (99.21%).
2. **Facial Reconstruction**: If a mask is detected, a Generative Adversarial Network (GAN)-based algorithm reconstructs the occluded facial features, effectively "removing" the mask virtually.
3. **Identity Recognition**: The reconstructed face is then processed by a deep convolutional neural network (CNN) to identify the individual by matching facial embeddings against a database.
4. **Emotion Detection**: Finally, the system analyzes the reconstructed face to detect and classify emotions such as happiness, sadness, anger, and more using an enhanced MobileNetV2 architecture.

## Key Features

- **High Accuracy**: The system achieves 99.21% accuracy in mask detection and robust performance in facial reconstruction and emotion recognition.
- **Real-Time Performance**: Designed to be lightweight and efficient, the system is suitable for real-time applications on mobile or embedded devices.
- **Robustness to Occlusion**: The facial reconstruction module ensures accurate identity and emotion recognition even when faces are partially obscured by masks.
- **Versatility**: The system can be applied in various domains, including security, healthcare, marketing, and human-computer interaction.

## Technologies Used

- **Deep Learning Frameworks**: TensorFlow, Keras, PyTorch
- **Computer Vision Libraries**: OpenCV, Dlib
- **Models**: MobileNetV2, GANs, U-Net, ResNet
- **Datasets**: CelebA, FER2013, CK+ (Cohn-Kanade)

## Repository Structure

- **/models**: Contains pre-trained models for mask detection, facial reconstruction, identity recognition, and emotion detection.
- **/datasets**: Includes scripts for downloading and preprocessing datasets.
- **/src**: Source code for the entire pipeline, including mask detection, facial reconstruction, identity recognition, and emotion detection modules.
- **/notebooks**: Jupyter notebooks for training and testing individual components of the system.
- **/docs**: Documentation and additional resources.

## Getting Started

To get started with the project, clone this repository and follow the instructions in the `README.md` file to set up the environment and run the system.

```bash
git clone https://github.com/your-username/facial-reconstruction-mask-removal.git
cd facial-reconstruction-mask-removal
```

facial reconstruction model = https://drive.google.com/file/d/1q0YN3RlhhEL0-RkfeKiMaAXrJomLR9og/view?usp=sharing
put it in = faceUnmask\models.

## Contributions

We welcome contributions from the community! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. For major changes, please discuss them first in the issues section.

## Acknowledgments

We would like to thank Yuan Ze University and our supervisor, Dr. Naeem Ul Islam, for their support and guidance throughout this project.
