import torchvision.models as torch_models
from torch import nn


def get_model(name, **kwargs):
    if name not in torch_models.__dict__:
        raise ValueError(f"Specified architecture '{name}' is not valid.")
    model = torch_models.__dict__[name](num_classes=1, **kwargs)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        out_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        out_dim = model.classifier.weight.shape[1]
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Cannot create {name} model. It does not have a Linear 'fc' or 'classifier' layer.")
    return model, out_dim
