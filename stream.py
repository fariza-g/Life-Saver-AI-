from flask import Flask, Response, render_template
import cv2
from twilio.rest import Client
import socket

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
    app.run(debug=True)
