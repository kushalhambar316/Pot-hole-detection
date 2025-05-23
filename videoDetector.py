import cv2
import os
import winsound  # Import winsound module for Windows

def detectPotholeonVideo(filename):
    # Reading the video file
    cap = cv2.VideoCapture(filename)

    # reading label name from obj.names file
    class_name = []
    with open(os.path.join("project_files", 'obj.names'), 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    # importing model weights and config file
    net1 = cv2.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
    net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model1 = cv2.dnn_DetectionModel(net1)
    model1.setInputParams(size=(640, 480), scale=1 / 255, swapRB=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analysis the stream with detection model
        classes, scores, boxes = model1.detect(frame, confThreshold=0.5, nmsThreshold=0.4)

        # Detection
        for (classid, score, box) in zip(classes, scores, boxes):
            label = "Pothole"
            x, y, w, h = box
            recarea = w * h
            area = frame.shape[1] * frame.shape[0]
            # Drawing detection boxes on frame for detected potholes and playing buzzer sound
            if (len(scores) != 0 and scores[0] >= 0.7):
                if ((recarea / area) <= 0.1 and box[1] < 600):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label, (box[0], box[1] - 10),
                               cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                    winsound.Beep(1500, 1000)  # Play buzzer sound
        cv2.imshow('Pothole Detection (Press Q to Close)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # End
    cap.release()
    cv2.destroyAllWindows()