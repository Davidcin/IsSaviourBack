virtual env:
python3 -m venv .venv
source .venv/bin/activate

install dependencies:
pip3 install flask flask-cors ultralytics opencv-python numpy

you won't see best practices today..but hey this shit was done for the demo so nu ganji da ar gangjian

---

app.py - simple flask endpoints for uploading processing and downloading videos, code is pretty self explanatory

video_process.py - simplified tennis_analyzed-YOLOv8, just to make it faster. you can try to play with the tennis analyzer main.py after the demo but this was done to speed up things. For future consider changing model, I didn't bother getting em from Roboflow, just used base medium model. here you can play with the confidence variable and etc.

my plan for this was to make this as easy to integrate as possible.
So haven't used that instant db (can be considered for the future in order to store this stuff somewhere) but right now my plan was just to host this back-end easily.

We could host this somewhere on cloud but quickest one to implement would be just to run api locally and use ngrok.

once you get url from ngrok just use that in front.
there are 4 endpoints:
GET /health - just to get 200 status code if api is up

POST /upload : in request body use form-data, in key write video choose file and attach videofile that you want to process - for uploading videos. in response you will get job id that you would need to pass as an url parameter in the next endpoint

POST /process/<job-id> - this is after uploading, it will process the video that has the same job id and will return link for downloading processed one, also returns some json data about the video itself.

POST /download/<job-id> - downloads processed video

uploaded videos will be stored in uploads dir
outputs dir will be storing already processed videos

## To run api just - python3 app.py

for improvements:
Host back somewhere
use tennis-analyzer not my simplified processor.
use custom trained models from Roboflow
instant db
