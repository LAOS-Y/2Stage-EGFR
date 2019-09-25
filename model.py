import torch
from torch import nn
from models import resnet


def generate_model(opt):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet10()
        elif opt.model_depth == 18:
            model = resnet.resnet18()
        elif opt.model_depth == 34:
            model = resnet.resnet34()
        elif opt.model_depth == 50:
            model = resnet.resnet50()
        elif opt.model_depth == 101:
            model = resnet.resnet101()
        elif opt.model_depth == 152:
            model = resnet.resnet152()
        elif opt.model_depth == 200:
            model = resnet.resnet200()
    
    return model
