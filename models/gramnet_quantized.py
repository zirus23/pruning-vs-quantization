import torch
from torchvision.models import resnet50
from models.gramnet import GramNet
from quantization.autoquant_utils import quantize_model
from quantization.base_quantized_model import QuantizedModel

class QuantizedGramNet(QuantizedModel):
    def __init__(self, gramnet, input_size=(1, 3, 224, 224), quant_setup=None, **quant_params):
        super().__init__(input_size)
        self.features = quantize_model(gramnet.features, **quant_params)
        self.feat_reducer = quantize_model(gramnet.feat_reducer, **quant_params)
        self.classifier = quantize_model(gramnet.classifier, **quant_params)

    def forward(self, x, extra_info=None):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.feat_reducer(x)
        if extra_info is not None:
            x = torch.cat((x, extra_info), dim=1)
        x = self.classifier(x)
        return x

def gramnet_quantized(pretrained=True, model_dir=None, load_type="fp32", num_class=8, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256, **qparams):
    gramnet = GramNet(resnet50(pretrained=False), 8)

    if load_type != "fp32":
        raise Exception("Unsupported load_type specified. Use 'fp32' and provide a model_dir.")

    ckpt = torch.load(model_dir)
    gramnet.load_state_dict(ckpt['model_state_dict'])
    # state_dict = torch.load(model_dir)
    # gramnet.load_state_dict(state_dict)

    return QuantizedGramNet(gramnet, **qparams)
