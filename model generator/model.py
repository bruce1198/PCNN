import torch
import numpy as np
import sys, os
from alexnet import Net
from os.path import abspath, dirname
pcnn_path = dirname(dirname(abspath(__file__)))
sys.path.insert(0, pcnn_path)


net = Net()
print(net)

model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
print(model)


with torch.no_grad():
    net.conv1.weight = model.features[0].weight
    net.conv2.weight = model.features[3].weight
    net.conv3.weight = model.features[6].weight
    net.conv4.weight = model.features[8].weight
    net.conv5.weight = model.features[10].weight
    net.conv1.bias = model.features[0].bias
    net.conv2.bias = model.features[3].bias
    net.conv3.bias = model.features[6].bias
    net.conv4.bias = model.features[8].bias
    net.conv5.bias = model.features[10].bias


    net.fc1.weight = model.classifier[1].weight
    net.fc2.weight = model.classifier[4].weight
    net.fc3.weight = model.classifier[6].weight
    net.fc1.bias = model.classifier[1].bias
    net.fc2.bias = model.classifier[4].bias
    net.fc3.bias = model.classifier[6].bias

from PIL import Image
from torchvision import transforms
filename = "C:\\Users\\zunzzz\\Desktop\\COMM\\images\\input.jpg"
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

net.load_state_dict(torch.load(os.path.join(pcnn_path, 'models', 'alexnet.h5')))

with torch.no_grad():
    output = net.forward_origin(input_batch)
    output_model = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
print(np.argmax(torch.nn.functional.softmax(output[0], dim=0).detach().numpy()))
print(np.argmax(torch.nn.functional.softmax(output_model[0], dim=0).detach().numpy()))
# torch.save(net.state_dict(), os.path.join(pcnn_path, 'models', 'alexnet.h5'))