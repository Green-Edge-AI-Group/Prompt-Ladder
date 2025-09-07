# import torch
# import clip
# from PIL import Image
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num,"Percentage":(trainable_num/total_num)*100}
# print(get_parameter_number(model.visual))#37828608 38131200
# print(model.visual)#86192640 86192640
# #
# # image = preprocess(Image.open("E:\COde\Master\MyCLIP-LADDER\\111.jpg")).unsqueeze(0).to(device)
# # text = clip.tokenize(["a diagram", "a dog", "a cat","a bird"]).to(device)
# #
# # with torch.no_grad():
# #     image_features = model.encode_image(image)
# #     text_features = model.encode_text(text)
# #
# #     logits_per_image, logits_per_text = model(image, text)
# #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# #
# # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
import torch
# import tensorboard
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda())
