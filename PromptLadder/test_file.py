import os.path as osp
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.pl1 import VisualEncoder,TextEncoder,LadderSide
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
_tokenizer = _Tokenizer()

def load_clip_to_cpu(name):
    backbone_name = name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

if __name__ == '__main__':
    device='cuda:0'

    # 验证图像模型是否正确
    clip_model = load_clip_to_cpu("ViT-B/16").to(device)
    image=torch.randn([1,3,224,224]).to(device)
    # image_feature1=clip_model.visual(image.type(clip_model.visual.conv1.weight.dtype))
    visual=VisualEncoder(clip_model).to(device)
    image_feature2,clip_image_feature=visual.forward(image.type(clip_model.visual.conv1.weight.dtype))
    # # print(layer_out)
    #
    # # 验证文本模型是否正确
    text=['a boy']
    text=clip.tokenize(text).to(device)
    # text_feature1=clip_model.encode_text(text)
    # text_feature2=TextEncoder(clip_model)(text)
    textencoder=TextEncoder(clip_model,1,1).to(device)
    text_feafure,clip_text_feature=textencoder(text)
    # print(text_layer_out)

    #验证side
    side_network=LadderSide(clip_model,3,[0,5,10]).to(device)
    # print(side_network)
    # print(id(a))
    # print(id(b))
    #
    # print(a.ln_2.data)
    print(clip_image_feature[0].shape)
    print(clip_text_feature[0].shape)
    # side_weight=side_network.ladders1.ln_2.state_dict()['weight']
    # clip_weight=clip_model.visual.transformer.resblocks[0].ln_2.state_dict()['weight']
    # print(side_weight)
    # print(clip_weight)
    # print(clip_weight.shape)
    print(side_network)
    # side_ladders1=copy.deepcopy(clip_model.visual.transformer.resblocks[0])
    # side_ladders1.ln_2.state_dict()['weight'][0]=2
    # print(side_ladders1.ln_2.state_dict()['weight'])
    # print(clip_model.visual.transformer.resblocks[0].ln_2.state_dict()['weight'])
    out=side_network(image.type(clip_model.visual.conv1.weight.dtype),text,clip_image_feature,clip_text_feature)

# b=side_network.ladders[0]
    # c=side_network.ladders1
    #
    # print(b.named_parameters())
