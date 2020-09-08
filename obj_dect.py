import cv2
import numpy as np 
import tensorflow.keras
from PIL import Image, ImageOps

np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')

cap = cv2.VideoCapture(0)
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while 1 > 0:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # image_array = np.asarray(img)
            # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # data[0] = normalized_image_array
            prediction = model.predict(data)
            print(prediction)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

