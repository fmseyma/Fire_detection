################################################################################
# Example : perform live fire detection in video using FireNet CNN
# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK
# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE
################################################################################

# This module detects fire
# env:fire_env

import cv2
import os
import sys
import math
import tensorflow as tf
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

args: list[int] = sys.argv

RTSP_URL = "rtsp://192.168.4.235:554/ch01.264"
RTSP_URL2 = "rtsp://192.168.4.190:554/ch01.264"

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
n = 0


def construct_firenet(x, y, training=False):
    # Build network as per architecture in [Dunnings/Breckon, 2018]
    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)
    network = conv_2d(network, 64, 5, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 1, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')

    if training:
        network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')

    if training:
        network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyper parameters
    if training:
        network = regression(network, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)

    # construct final model
    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model


if __name__ == '__main__':

    # construct and display model
    realTriggerTime = 2
    triggerTime = int(realTriggerTime * 5 / 3)
    model = construct_firenet(224, 224, training=False)
    print("Constructed FireNet ...")
    model.load(os.path.join("models/FireNet", "firenet"), weights_only=True)
    print("Loaded CNN network weights ...")

    # network input sizes
    rows = 224
    cols = 224

    # display and loop settings
    windowName = "Live Fire Detection - FireNet CNN"
    keepProcessing = True

    if len(args) == 2:
        # load video file from first command line argument
        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")
        print(video.isOpened())
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        # get video properties
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        videoWriter = cv2.VideoWriter('output.avi', fourcc, 24, (frame_width, frame_height))

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000 / fps)
        if fps == 0:

            triggerTime = int(triggerTime * 20)
            realTriggerTime = int(realTriggerTime * 20)

        else:
            triggerTime = int(triggerTime * fps)
            realTriggerTime = int(realTriggerTime * fps)

        counter = int(triggerTime + 1)
        fireCounter = 0

        while keepProcessing:

            # start a timer (to see how long processing and display takes)
            start_t = cv2.getTickCount()

            # get video frame from file, handle end of file
            ret, frame = video.read()
            if ret:

                cv2.putText(frame, 'fps: ' + str("%.2f" % fps), (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                if counter % triggerTime == 0:
                    counter = 0
                    print(str("%.2f" % fps))

                    if fireCounter >= realTriggerTime:
                        print("real fire")
                        cv2.imwrite(f'../fire_detection/images/image_{n}.png', frame)

                        n = n + 1
                    fireCounter = 0
                counter += 1

                # re-size image to network input size and perform prediction
                small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

                '''
                perform prediction on the image frame which is:
                - an image (tensor) of dimension 224 x 224 x 3
                - a 3 channel colour image with channel ordering BGR (not RGB)
                - un-normalised (i.e. pixel range going into network is 0->255)
                '''

                output = model.predict([small_frame])

                # label image based on prediction
                if round(output[0][0]) == 1:
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 50)
                    cv2.putText(frame, 'FIRE', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    fireCounter += 1
                else:
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 0), 50)
                    cv2.putText(frame, 'CLEAR', (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # stop the timer and convert to ms. (to see how long processing and display takes)
                stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000

                # image display and key handling
                cv2.imshow(windowName, frame)
                videoWriter.write(frame)

                # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
                key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF

                if key == ord('q'):
                    keepProcessing = False
                    videoWriter.release()
                elif key == ord('f'):
                    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("FULL")
