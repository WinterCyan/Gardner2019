import torch
import urllib
from PIL import Image
from torchvision import transforms
import urllib

model_dir = 'C:\\code\\Gardner2019\\Files\\densenet121.pth'
if __name__ == '__main__':
    # model = torch.load(model_dir)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    model.eval()
    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # try: urllib.URLopener().retrieve(url, filename)
    # except: urllib.request.urlretrieve(url, filename)
    input_img = Image.open('../Files/dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_img)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda')
    print(input_batch.shape)
    model.to('cuda')

    output = model(input_batch)
    print(output[0].shape)
    print(torch.argmax(output[0]))
    dict = model.state_dict()
    for k, v in dict.items():
        print(k)
