import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
import json


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
DEVICE = torch.device("cuda")
print(DEVICE)

CLASSES = json.load(open("classes.txt"))

def close_stream(vid):
    vid.release()
    cv2.destroyAllWindows()

def frame_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    return torch.FloatTensor(frame)

def get_objects_in_tensor(tensor,model):
    tensor = tensor.to(DEVICE)
    outputs = model(tensor)
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()
    for index in range(len(scores)):
        if scores[index] < 0.8:
            return (boxes[:index],labels[:index],scores[:index])

def draw_objects(orig_frame,boxes,labels,scores):
    for index in range(len(scores)):
        box = boxes[index]
        label = labels[index]
        score = scores[index]
        (startX, startY, endX, endY) = box.astype("int")
        orig_frame = cv2.rectangle(orig_frame, (startX, startY), (endX, endY),(255, 0, 0),2)
        orig_frame = cv2.putText(orig_frame, CLASSES[str(label)] + " " + str(score),(startX, startY),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0))
    cv2.imshow('frame',orig_frame )

def run_main():
    vid = cv2.VideoCapture(0)
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, progress=False).to(DEVICE)
    model = model.eval()
    
    while(True):
        ret, frame = vid.read()
        orig_frame = frame.copy()
        
        tensor = frame_to_tensor(frame)
        (boxes,labels,scores) = get_objects_in_tensor(tensor,model)
        draw_objects(orig_frame, boxes,labels,scores)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    close_stream(vid)
run_main()