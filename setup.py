import torch
import os


input = "input"
output = "output"

if not os.path.exists(input):
    os.mkdir("input")
if not os.path.exists(output):
    os.mkdir("output")

model = torch.hub.load("PeterL1n/RobustVideoMatting",
                       "mobilenetv3").cuda()  # or "resnet50"
convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
