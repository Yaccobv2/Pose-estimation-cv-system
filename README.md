# Pose-estimation-for-weightlifters-cv-system
Pose estimation for weightlifters cv system

# Purpose of the project

The goal of the project is to create a vision system capable of detecting the barbell and the athlete's foot during exercise in order to accurately analyze movement quality.


# Used tools
* darknet framework
* yolov4/yolov4-tiny
* google colabolatory
* python
* medipipe
* opencv2
* numpy

# Training yolov4 and yolov4-tiny

<img src="https://user-images.githubusercontent.com/39679208/172713616-990ca93d-d562-41db-9b12-cc30a0a5259c.png" width="400" height="400" />

Yolov4 achieved a mAp of about 99%.

<img src="https://user-images.githubusercontent.com/39679208/172713878-5b267673-28ad-48c3-9036-9c165f05a4ae.png" width="400" height="400" />

Yolov4-tiny achieved a mAp of about 91%.

# Pose detection module
The publicly available mediapipe library was used for position estimation. The library is based on neural networks, and the solutions implemented in it do not require training.  The neural network outputs a set of points specifying the location of such body parts as eyes, mouth, nose, shoulders, elbows, hands, hip spikes, knees and feet and ankles. Using this information, the user is able to effectively determine joint angles and body alignments.
A separate module called "plotter_3d.py" was created to visualize the output from the pose detector. It allows the user to save an animation containing an image and a 3d plot of the recorded person's pose after the detection is complete.

<img src="https://user-images.githubusercontent.com/39679208/172714361-fee89bf8-7b28-4f8b-a137-cb9e221db38e.png" width="500" height="300" />
<img src="https://user-images.githubusercontent.com/39679208/172714391-4eaad3ee-fbe0-4a85-9c88-770ad0fa4b0f.png" width="500" height="300" />
<img src="https://user-images.githubusercontent.com/39679208/172714519-27e261cf-e7f4-4093-b943-c32d47b06776.png" width="500" height="300" />
Unfortunately, pose detection during exercise proved to be impossible because the weight on the barbell covered parts of the exerciser's body, making the algorithm unable to perform detection.


# Implementation 
This project was implemented on the jetson-nano development kit. We evaluated it using gpu acceleration using OpenCV compiled with CUDA and Cudnn. The system achieved about 7 frames per second when detecting the foot and the barbell. 


# Video presentation
[![VIDEO](https://user-images.githubusercontent.com/39679208/172718905-93d761bb-e7b2-449c-8d88-adba6a132a69.png)](https://youtu.be/TkXbZyNji7U)


# How to run this project
To run this project, create a virtualenv with python, preferably version 3.9, and install the necessary libraries using: ``pip install requirements.txt`.
If you want to achive better performance compile OpenCV with CUDA, you can follow this tutorial: https://www.youtube.com/watch?v=YsmhKar8oOc. 
Now you can type ``python main.py`` in python and run the program with the desired flags defined in the table below.


| Shortened argument flag | Full flag of an argument | Description |
| :---: | :---: | :---: |
| -s | --save | Saves data and graphs when recording is complete. |
| -vp | --video-path | Specifies the path to the recorded video. If the path is not specified, the detection will start from camera port 0. |
| -pd | --pose-detector | Performs pose detection using the mediapipe module. |
| -a | --angles | Computes angles at the joints specified in the constants file. (Module available only with the -pd argument.) |
| -b | --barbell-detector | Performs barbell and foot detection. |
| -f | --fps | Displays the number of fps. |
| -sa | --save-animation | Saves the animation after the detection. Specify the format. Available formats: mp4. (Module available only with the -pd argument.)|
| -d | --display | Displays a preview during recording. |
| -o | --output-file | Specifies the storage location and name of the csv file. |
