"""
CameraFaceCapture
"""
import cv2
from datetime import datetime

BASE_TIMEOUT = 3
DEFAULT_FILE_NAME = "out.avi"


class CameraFaceCapture:
    """
    CameraFaceCapture:

    This class creates a camera capture session and records whenever a face is detected
    """

    def __init__(self, out_file_path=DEFAULT_FILE_NAME, display_camera=True):
        self._face_cascade = cv2.CascadeClassifier("face_cascade.xml")
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            print("Cannot open camera")
            exit()
        self._video_writer = self.init_video_writer(out_file_path=out_file_path)
        self._capturing = False
        self._timestamp = 0
        self._display_camera = display_camera

    def init_video_writer(self, out_file_path):
        frame_width = int(self._cap.get(3))
        frame_height = int(self._cap.get(4))

        out = cv2.VideoWriter(out_file_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                              (frame_width, frame_height))
        return out

    def loop(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self._cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_DO_ROUGH_SEARCH
            )

            # determine if capture is turned on or off.
            if len(faces) > 0:
                self._capturing = True
                self._timestamp = datetime.now().timestamp()
            elif self._capturing:
                now_ts = datetime.now().timestamp()
                # stop capturing after a buffer timeout where no faces were detected.
                if self._timestamp + BASE_TIMEOUT < now_ts:
                    capturing = False

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self._capturing:
                self._video_writer.write(frame)

            # Display the resulting frame
            if self._display_camera:
                cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture
        self._video_writer.release()
        self._cap.release()
        cv2.destroyAllWindows()


# Test function
if __name__ == "__main__":
    cam = CameraFaceCapture()
    cam.loop()
