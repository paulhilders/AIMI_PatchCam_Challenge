import torch
from PIL import Image
import numpy
import sys
# from torchvision import transforms
import numpy as np
import cv2

import matplotlib.pyplot as plt
import Dataloader
from vit_pytorch import ViT
import timm

def rollout(attentions, discard_ratio, head_fusion):
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*I)/2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.model.eval()
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)

def show_mask_on_image(img, mask):
    if keras_test:
        img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == "__main__":
    print("Started loading Dataloaders...")
    train_dataloader, test_dataloader, valid_dataloader = Dataloader.getDataLoaders()
    print("Dataloaders loaded")

    keras_test = False
    if keras_test:
        from PIL import Image
        import glob
        image_list = []
        for filename in glob.glob('FailedKerasSamples/*.jpg'):
            im=np.array(Image.open(filename))
            image_list.append(im)

    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2, img_size=96)

    model.eval()
    model.load_state_dict(torch.load('./models/ViT_pretrained_DA.pth', map_location=torch.device('cpu')),strict=False)

    if keras_test:
        for index, image in enumerate(image_list):
            image = torch.from_numpy(image)
            att_rollout = VITAttentionRollout(model)
            mask = att_rollout(image.permute(2,1,0).float().unsqueeze(0))
            np_img = np.array(image)[:,:,::-1]
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
            mask = show_mask_on_image(np_img, mask)
            plt.imshow(mask)
            plt.axis('off')
            # plt.savefig(f'vit_{index}', bbox_inches='tight', pad_inches=0)
            plt.show()
    else:
        _, input_tensor = next(iter(enumerate(test_dataloader)))
        input_image = input_tensor["image"]
        true_label = input_tensor["label"]
        for index, (image, label) in enumerate(zip(input_image, true_label)):
            img = image.permute(1,2,0)
            att_rollout = VITAttentionRollout(model)
            mask = att_rollout(image.unsqueeze(0).float())
            np_img = np.array(img)[:,:,::-1]
            mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0])) # ????
            mask = show_mask_on_image(np_img, mask)
            plt.imshow(mask)
            plt.axis("off")
            # plt.savefig(f'vit_DA_{index}', bbox_inches='tight', pad_inches=0)
            plt.show()