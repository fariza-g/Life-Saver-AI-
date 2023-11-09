from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from twilio.rest import Client
import serial
from flask import Flask, Response, render_template
import socket

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize the list of class labels MobileNet SSD was trained to detect
# and generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()


# Initialize the Twilio client
def initialize_twilio():
    account_sid = 'AC0d4c7009b3df18d01f275a03e7be4966'
    auth_token = '9f2d63acd0b6c0c49c1f8a307d26fe50'
    return Client(account_sid, auth_token)


twilio_client = initialize_twilio()

# Initialize a flag to keep track of whether a message has been sent
message_sent = False

# Initialize the time of detection
time_of_detection = None


# Function to send an SMS
def sending_sms(text='WARNING...', receiver='+77754600667'):
    try:
        message = twilio_client.messages.create(
            body=text,
            from_='+14243295391',
            to=receiver
        )
        print('Message sent:', message.sid)

        # Send command '1' to Arduino to trigger the speaker
        with serial.Serial('/dev/tty.usbmodem1101', 9600) as ser:  # Adjust the serial port as needed
            ser.write(b'1')

        return 'The message was successfully sent!'
    except Exception as ex:
        print('Something was wrong... :', ex)
        return 'Something was wrong... :', ex


# Function to send live stream link
def send_live_stream_link():
    app = Flask(__name__)
    camera = cv2.VideoCapture(0)

    # Initialize the Twilio client
    account_sid = 'AC0d4c7009b3df18d01f275a03e7be4966'
    auth_token = '9f2d63acd0b6c0c49c1f8a307d26fe50'
    client = Client(account_sid, auth_token)

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video')
    def video():
        # Get the server's IP address
        server_ip = socket.gethostbyname(socket.gethostname())

        # Replace 'YOUR_NGROK_URL' with your actual ngrok URL
        live_streaming_link = "https://d70c-89-218-213-178.ngrok-free.app"

        # Send an SMS with the link
        receiver = '+77754600667'  # Replace with the recipient's phone number
        message = f"Watch the live stream at {live_streaming_link}"

        try:
            message = client.messages.create(
                body=message,
                from_='+14243295391',
                to=receiver
            )
            print(f"Message sent: {message.sid}")
        except Exception as ex:
            print(f"Something went wrong: {ex}")

        # Return the live streaming page
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    if __name__ == '__main__':
        app.run(debug=True, port=8080)


# Main loop
while True:
    # Grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # Pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # Extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == 'person' and not message_sent:
                if time_of_detection is None:
                    # Set the time of detection when a person is first detected
                    time_of_detection = time.time()
                elif (time.time() - time_of_detection) >= 5:  # Send SMS 5 seconds after detection
                    # Send an SMS after 5 seconds
                    text = 'A person has been detected!'
                    receiver = '+77754600667'  # Replace with the recipient's phone number
                    sending_sms(text, receiver)

                    # Send the live stream link
                    send_live_stream_link()
                    message_sent = True

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()