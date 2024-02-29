import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
import torchvision.models as models
from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_model import QuantizedModel
from torch.ao.quantization import quantize_dynamic
import torchvision

import torch
import numpy as np
from torch import nn
from torchvision.models import resnet50
from torchvision import transforms, datasets

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets

model_dir = "/home/user/pruning-vs-quantization/gramood/resnet-50_checkpoint.pth"
skin_train_dir = "/home/user/pruning-vs-quantization/gramood/data/skin_cancer/train"
skin_test_dir = "/home/user/pruning-vs-quantization/gramood/data/skin_cancer/test"

class GramNet(nn.Module):
    """
    This class gets a resnet model, changes its FC layer and include the extra information on it (if applicable).
    """

    def __init__(self, resnet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256):
        super(GramNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        self.feat_maps = []
        self.feat_reducer = nn.Sequential(
            nn.Linear(2048, neurons_class),
            nn.BatchNorm1d(neurons_class),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
        )
        self.classifier = nn.Linear(neurons_class + n_extra_info, num_class)

    def forward(self, img, extra_info=None):
        x = self.features(img)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.feat_reducer(x)
        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x
        return self.classifier(agg)

    def _hook_fn(self, m, i, o):
        self.feat_maps.append(o)

    def hook_layers(self):
        for layer in self.features.modules():
            if isinstance(layer, models.resnet.Bottleneck):
                layer.register_forward_hook(self._hook_fn)

    def evaluate(self, data_loader, device="cuda"):
        self.eval()
        self.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

def main():
    batch_size = 30
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    sk_test = torch.utils.data.DataLoader(
        datasets.ImageFolder(skin_test_dir, transform=trans),  # Update your path
        batch_size=batch_size,
    )

    resnet_model = models.resnet50(pretrained=False)
    model = GramNet(resnet_model, num_class=8)  # Set your number of classes
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # TODO: add this to the class itself
    model.hook_layers()

    accuracy = model.evaluate(sk_test, device="cuda")
    print(f'Accuracy: {accuracy}%')

class QuantizedGramNet(GramNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = quantize_dynamic(self.features)
        self.feat_reducer = quantize_dynamic(self.feat_reducer)
        self.classifier = quantize_dynamic(self.classifier)

    def forward(self, img, extra_info=None):
        x = self.features(img)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.feat_reducer(x)
        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x
        res = self.classifier(agg)
        return res


def gramnet_quantized(pretrained=True, num_class=8, load_type="fp32", model_dir=None):
    if load_type != "fp32":
        raise Exception(
            "Unsupported load_type specified. Use 'fp32' and provide a model_dir."
        )

    # Load the model
    model = models.resnet50(pretrained=pretrained)
    gramnet = GramNet(model, num_class)

    if model_dir is not None:
        gramnet.load_state_dict(torch.load(model_dir)["model_state_dict"])

    # Instantiate QuantizedGramNet with the pretrained GramNet
    return QuantizedGramNet(gramnet)


def main_2():
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    model_dir = "/home/user/pruning-vs-quantization/gramood/resnet-50_checkpoint.pth"

    num_classes = 8
    pretrained_resnet = resnet50(pretrained=False)
    gramnet = GramNet(resnet=pretrained_resnet, num_class=num_classes)
    ckpt = torch.load(model_dir)
    gramnet.load_state_dict(ckpt['model_state_dict'], strict=True)

    from utils.skincancer_dataloaders import SkinCancerDataLoaders
    from torchvision.transforms import InterpolationMode

    dataloaders = SkinCancerDataLoaders(
        "/home/user/pruning-vs-quantization/gramood/data/skin_cancer",
        224,
        128,
        4,
        InterpolationMode.BILINEAR,
    )

    # from utils.imagenet_dataloaders import ImageNetDataLoaders
    # dataloaders = ImageNetDataLoaders(
    #     "/home/user/pruning-vs-quantization/gramood/data/imagenet",
    #     224,
    #     128,
    #     4,
    #     InterpolationMode.BILINEAR,
    # )

    accuracy = gramnet.evaluate(
        dataloaders.val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"Accuracy of the quantized model: {accuracy}%")

if __name__ == "__main__":
    main()
