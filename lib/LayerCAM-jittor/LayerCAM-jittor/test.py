import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
import argparse
from utils import *
from cam.layercam import *
from ViT import VisionTransformer

jt.flags.use_cuda = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='The Pytorch code of LayerCAM')
    parser.add_argument('--img_path', type=str, default='../../data/test/60/image_06247.jpg', help='Path of test image')
    parser.add_argument('--layer_id', type=list, default=['blocks'], help='The cam generation layer') #, 9, 16, 23, 30
    # #,['patch_embed', 'pos_drop', 'blocks', 'norm', 'head']
    # images/ILSVRC2012_val_00000476.JPEG
    return parser.parse_args()
    
if (__name__ == '__main__'):
    args = get_arguments()
    input_image = load_image(args.img_path)
    input_ = apply_transforms(input_image)
    print(input_.shape)

    #======================== ViT ==============================
    vitModel = VisionTransformer(img_size=224,patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,num_classes=1020)
    vitModel.load("../../model/ViT/finetuned_2_2.pkl")
    # print(vitModel)
    print(vitModel._modules['blocks'][0])
    # fsdhkj
    optimizer = nn.SGD(vitModel.parameters(), 0.1)
    for i in range(len(args.layer_id)):
        layer_name = (str(args.layer_id[i])) #'features_' + 
        vit_model_dict = dict(type='ViT', arch=vitModel, layer_name=layer_name, input_size=(224, 224))
        vit_layercam = LayerCAM(vit_model_dict, optimizer)
        predicted_class = vitModel(input_).max(1)[(- 1)]
        layercam_map = vit_layercam(input_)
        print(layercam_map.numpy())
        basic_visualize(input_.numpy(), layercam_map.numpy(), save_path='./vis/image_06247_3.png')#.format((i + 1)

    
    # GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    
    # optimizer = nn.Adam(model.parameters(), lr=learningRate, weight_decay=1e-3)
    # scheduler = CosineAnnealingLR(optimizer, 15, eta_min=etaMin)

    # vgg = models.vgg16(pretrained=True)
    # optimizer = nn.SGD(vgg.parameters(), 0.1)


    # for i in range(len(args.layer_id)):
    #     layer_name = ('features_' + str(args.layer_id[i]))
    #     vgg_model_dict = dict(type='vgg16', arch=vgg, layer_name=layer_name, input_size=(224, 224))
    #     vgg_layercam = LayerCAM(vgg_model_dict, optimizer)
    #     predicted_class = vgg(input_).max(1)[(- 1)]
    #     layercam_map = vgg_layercam(input_)
    #     basic_visualize(input_.numpy(), layercam_map.numpy(), save_path='./vis/stage_{}_1.png'.format((i + 1)))