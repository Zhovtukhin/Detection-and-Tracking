import cv2
import sys
import numpy as np

# Initialize the parameters
objectnessThreshold = 0.5  # Objectness threshold
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.1  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf#sports ball

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    ''''# Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)'''

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold :
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    bboxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        classId = int(classIds[i])
        label = ''
        if classes:
            assert (classId < len(classes))
            label =  classes[classId]
        if label == 'sports ball':
            bboxes.append([confidences[i],classIds[i], left, top, width, height])
    #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    boxes = np.array(bboxes)
    if bboxes != []:
        return tuple((boxes[np.argsort(boxes[:,0])][0, 2:]).astype(int))
    return ()

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'CSRT', 'MOSSE']
tracker_type = tracker_types[6] # [int(sys.argv[1])]


def initTracker():
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
            print(t)
    return tracker

if __name__ == '__main__' :

    # Set up tracker.
    # Choose one tracker




    # Read video
    filename = "soccer-ball.mp4"
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    # Define an initial bounding box

    tracker = initTracker()

    classesFile =   "models/coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration =  "models/yolov3.cfg"
    modelWeights = "models/yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    bbox = postprocess(frame, outs)
    if not bbox:
        bbox = (561, 316, 117, 115)

    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 178, 50), 2, cv2.LINE_AA)

    # Uncomment the line below to select a different bounding box
    # bbox = cv2.selectROI(frame, False)
    print("Initial bounding box : {}".format(bbox))
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    out = cv2.VideoWriter('{}_{}_{}.mp4'.format(video_name,tracker_type,bbox),cv2.VideoWriter_fourcc(*'MP4V'), 30, (640,360))
    failCount = 50
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        if failCount % 100 == 0 and tracker_type != tracker_types[3]:
            ok = False

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv2.LINE_AA)

            if failCount >= 50 and tracker_type != tracker_types[3]:
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
                net.setInput(blob)
                outs = net.forward(getOutputsNames(net))
                bbox = postprocess(frame, outs)
                if bbox:
                     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 178, 50), 2,
                              cv2.LINE_AA)
                     tracker = initTracker()
                     ok = tracker.init(frame, bbox)
                else:
                    failCount = 0
            failCount += 1

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2, cv2.LINE_AA);
        #print(fps)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA);


        # Display result
        cv2.imshow("Tracking", frame)

        outframe = cv2.resize(frame, (640,360))
        out.write(outframe)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    out.release()

    video.release()