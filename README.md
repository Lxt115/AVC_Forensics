# AVC_Forensics: Face Forgery Detection System for Adversarial Attacks

## Project Objectives

The main objectives of our project are focused on the following three aspects:

1. Fusion of Audio-Visual Features: While lip features and audio signals can be used as separate detection methods, it is advantageous to combine them as a basis for detection.

2. Improved Resistance to Interference: By freezing lip features in time and applying them to audio-visual consistency forgery detection, the accuracy of detection can be significantly enhanced, even in the presence of high levels of noise.

3. Enhanced Security Performance: Develop a multimodal detection model that can withstand adversarial attacks on the samples.
   
## Pipline
![image](https://github.com/Lxt115/AVC_Forensics/blob/main/image/pipline.png)

## Website
<img src="https://github.com/Lxt115/AVC_Forensics/blob/main/image/ui2.png" width="500"/><img src="https://github.com/Lxt115/AVC_Forensics/blob/main/image/ui3.png" width="500"/>

<img src="https://github.com/Lxt115/AVC_Forensics/blob/main/image/ui1.png" width="500"/>

## Dataset
1.DFDC: The DFDC dataset is a large-scale audio-visual fusion dataset created by Facebook for the Deepfake Detection Challenge. It consists of over 100,000 clips from 3,426 paid actors, incorporating 8 deepfake methods and 19 interference techniques. Models trained on this dataset can generalize well to real-world deepfake videos.

2.FaceForensics++: This dataset includes 1,000 video sequences processed with automatic face manipulation methods such as Deepfakes, Face2Face, FaceSwap, and NeuralTextures. It is primarily used for evaluating detection algorithms. The videos have low overall quality and exhibit noticeable artifacts from the manipulations.



## Implementation
### Visual Network:
   - Extract lip images corresponding to each frame of the video file, based on the Lipforensics paper.
   - Utilize 3D convolutional layers and residual networks to extract comprehensive lip feature sequences. The purpose of the residual network is to prevent gradient explosion and extract deep-level features.
   - Input the features into a multi-scale temporal convolutional network for lip-reading training, leveraging the irregularities in mouth movements between genuine and fake videos for deepfake detection.

### MS-TCN (Multi-Scale Temporal Convolutional Network):
   - Based on the TCN (Temporal Convolutional Network), enhance it by providing multiple time scales to mix short-term and long-term information during feature encoding.
   - Each time convolution is composed of several branches with different kernel sizes, and their outputs are combined through simple concatenation. This allows each convolutional layer to mix information at multiple time scales.

### Speech Recognition:
   - Extract audio features from the video file and input them into a residual network for deep convolution.
   - Employ two layers of Bidirectional Gated Recurrent Units (BGRU) to model the temporal dynamic features of the audio signal, which will be used for deepfake detection based on the temporal inconsistency between audio and visual inputs.

### Fusion Processing:
   - After processing through the visual and auditory networks, the original video is transformed into image feature sequences with temporal dynamics and audio feature streams.
   - Fusion of the video and audio features is performed using a fusion token-based method, followed by deep fusion through a Transformer layer.
   - Finally, a softmax classifier is used for binary classification to achieve the desired prediction.
     
We conducted performance testing of the model in four aspects: Accuracy, Generalization, Robustness, and Security.
