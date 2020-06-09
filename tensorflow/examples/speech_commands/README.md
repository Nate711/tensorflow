# Online Command Word Learning

Run ```train_tf2.py``` to train a small Tensorflow 2 model to recognize the the words in the Speech Commands Dataset.

Then run ```transfer_learn.py``` to record voice samples from your computer to learn a name online. After listening and learning, the code runs inference continually. 

The code is designed to interface with the Stanford Pupper robot over a UDPComms socket, port number 8008. When the robot's name is heard, the string ```"name"``` is published to the socket, while ```"noise"``` is published otherwise.

## Requirements

Python:
- UDPComms
- PyAudio

Other:
- PortAudio