import torch
from torch import nn
from torchvision.models import resnet50
from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_model import QuantizedModel

class GramNet(nn.Module):
    """
    This class get a resnet model, changes its FC layer and include the extra information on it (if applicable)
    """

    def __init__(self, resnet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None):
        """

        :param resnet: the resnet model from torchvision.model. Ex: model.resnet50
        :param num_class: the number of task's classes
        :param freeze_conv: if you'd like to freeze the extraction map from the model
        :param n_extra_info: the number of extra information you wanna include in the model
        :param p_dropout: the dropout probability
        """
        super(GramNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if feat_reducer is None:
            self.feat_reducer = nn.Sequential(
                nn.Linear(2048, neurons_class),
                nn.BatchNorm1d(neurons_class),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.feat_reducer = feat_reducer

        # Here comes the extra information (if applicable)
        if classifier is None:
            self.classifier = nn.Linear(neurons_class + n_extra_info, num_class)
        else:
            self.classifier = classifier

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

class QuantizedGramNet(QuantizedModel):
    def __init__(self, gramnet, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)
        self.features = quantize_model(gramnet.features, **quant_params)
        self.feat_reducer = quantize_model(gramnet.feat_reducer, **quant_params)
        self.classifier = quantize_model(gramnet.classifier, **quant_params)

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

def gramnet_quantized(pretrained=True, model_dir=None, load_type="fp32", num_class=8, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256, **qparams):
    gramnet = GramNet(resnet50(pretrained=True), num_class, freeze_conv, n_extra_info, p_dropout, neurons_class)

    if load_type != "fp32":
        raise Exception("Unsupported load_type specified. Use 'fp32' and provide a model_dir.")

    ckpt = torch.load(model_dir)
    gramnet.load_state_dict(ckpt['model_state_dict'], strict=True)
    return QuantizedGramNet(gramnet, **qparams)

def main():
    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    pretrained_resnet = resnet50(pretrained=False)

    num_classes = 8
    gramnet = GramNet(resnet=pretrained_resnet, num_class=num_classes)
    ckpt = torch.load("/home/user/pruning-vs-quantization/gram-ood/resnet-50_checkpoint.pth")
    gramnet.load_state_dict(ckpt['model_state_dict'], strict=True)

    # Function to evaluate the model
    def evaluate_model(model, data_loader, device="cuda"):
        model.eval()
        model.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    from utils.skincancer_dataloaders import SkinCancerDataLoaders
    from torchvision.transforms import InterpolationMode
    dataloaders = SkinCancerDataLoaders(
        "/home/user/pruning-vs-quantization/gram-ood/data/skin_cancer",
        224,
        128,
        4,
        InterpolationMode.BILINEAR,
    )

    accuracy = evaluate_model(gramnet, dataloaders.val_loader, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f'Accuracy of the unquantized model: {accuracy}%')

if __name__ == "__main__":
    main()
