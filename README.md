# Urban Sound classification
For the time being: binary classifier for classes dog bark and engine_idling
- fe.py extracts features
- aeai.py trains and saves the model
- test.py runs the model on the testing set 
- passtoAI.py passes a wav file into the AI
- server.py starts a web server that you can use to pass files to the AI. Only works under Linux.
 
Please note that if you want to test it, input file needs to be resampled to 48000 and cropped - up to 4 seconds. Use ffmpeg.

