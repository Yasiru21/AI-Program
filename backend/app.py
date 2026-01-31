from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import webbrowser
import threading

#Initialize the Flask app
app = Flask(__name__)

#Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

#Open webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raiseIOError("Cannot open webcam") #pyright: ignore[reportUndefinedVariable]
except IOError as e:
    print(f"Error: {e}")

def generate_frames():
    """
    This function reads frames from the webcam, runs YOLO detection,
    and yields them as JPEG-encoded bytes for streaming.
    """
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Ending stream.")
            break

        # Perform detection
        results = model(frame)

        #Plot results on the frame
        annotated_frame = results[0].plot()

        #Encode the frame as JPEG
        (flag, encoded_image) = cv2.imencode('.jpg', annotated_frame)
        if not flag:
            continue #Skip frame if encoding fails

        #Convert the encoded image to bytes
        frame_bytes = encoded_image.tobytes()

        #Yield the frame in the required multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serves the main HTML page."""
    #looks for 'index.html' in the 'templates' folder
    return render_template('index.html')

@app.route('/video')
def video():
    #Provides the video stream.
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    #Opens the default web browser to the app URL.
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Run browser in a separate thread to avoid blocking Flask
    threading.Timer(1.5, open_browser).start()
    
    # Runs the Flask app
    # host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
    