from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.checkpoint as cp
from collections import OrderedDict
from Networks.DenseNet import *


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # # todo: rewrite when torchscript supports any
    # def any_requires_grad(self, input):
    #     # type: (List[Tensor]) -> bool
    #     for tensor in input:
    #         if tensor.requires_grad:
    #             return True
    #     return False

    # @torch.jit.unused  # noqa: T484
    # def call_checkpoint_bottleneck(self, input):
    #     # type: (List[Tensor]) -> Tensor
    #     def closure(*inputs):
    #         return self.bn_function(inputs)
    #
    #     return cp.checkpoint(closure, *input)

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class ParamLENet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, latent_size=3072, decode_size=512, num_lights=3, memory_efficient=False):

        super(ParamLENet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # headless ?
        # # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        #
        # # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # latent vector layer
        # self.features.add_module('latent', nn.Linear(num_features, latent_size))
        self.latent = nn.Linear(num_features, latent_size)

        # decode_layer
        # self.features.add_module('decode', nn.Linear(latent_size, decode_size))
        self.decoder = nn.Linear(latent_size, decode_size)

        # Official init from torch repo.
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, 0)

        self.d_out = nn.Linear(decode_size, num_lights)
        self.l_out = nn.Linear(decode_size, 3*num_lights)
        self.s_out = nn.Linear(decode_size, num_lights)
        self.c_out = nn.Linear(decode_size, 3*num_lights)
        self.a_out = nn.Linear(decode_size, 3)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # output: 1*1, average pooling over full feature map
        out = torch.flatten(out, 1)  # flatten: keep some dims and merge the others
        #  replace the classifier with two FC layers
        latent_vec = self.latent(out)
        decode_vec = self.decoder(latent_vec)
        d = self.d_out(decode_vec)
        l = self.l_out(decode_vec)
        s = self.s_out(decode_vec)
        c = self.c_out(decode_vec)
        a = self.a_out(decode_vec)
        return d, l, s, c, a


if __name__ == '__main__':
    pretrained_densenet121 = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    pretrained_densenet121_dict = pretrained_densenet121.state_dict()
    paramlenet = ParamLENet()
    init_paramlenet = ParamLENet()
    paramlenet_dict = paramlenet.state_dict()
    shared_weights = {k:v for k, v in pretrained_densenet121_dict.items() if k in paramlenet_dict}
    paramlenet_dict.update(shared_weights)
    paramlenet.load_state_dict(paramlenet_dict)

    # print(len(pretrained_dict.items()))
    # print(len(my_dict.items()))
    # print(init_net121.state_dict()['features.denseblock4.denselayer16.norm2.weight'])
    # print(net121.state_dict()['features.denseblock4.denselayer16.norm2.weight'])
    # print(init_net121.state_dict()['classifier.bias'])
    # print(net121.state_dict()['classifier.bias'])
    # print('---------pretrained----------')
    # for k, v in pretrained_dict.items():
    #     print(k)
    # print('---------my----------')
    # for k, v in my_dict.items():
    #     print(k)

    # input_img = Image.open('../Files/dog.jpg')
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # input_tensor = preprocess(input_img)
    # input_batch = input_tensor.unsqueeze(0)
    # input_batch = input_batch.to('cuda')
    # paramlenet.to('cuda')
    # d, l, s, c, a = paramlenet(input_batch)
    # init_paramlenet.to('cuda')
    # d2, l2, s2, c2, a2 = init_paramlenet(input_batch)
    # print(d, l, s, c, a)
    # print(d2, l2, s2, c2, a2)
