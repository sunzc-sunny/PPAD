import sys
sys.path.append('..')
import torch
import numpy as np
import sklearn.metrics as metrics
from PIL import Image
from ppad_clip.datasets.mask_vincxr import MaskVinCXR
from torchvision import transforms
from ppad_clip.model import convert_weights
from ppad_clip.ppad import PPAD
import warnings
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import random
import datetime



warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

STATUS = ['normal', 'pneumonia']

# pre-trained model path
model_path = "./best_64_0.0001_original_35000_0.864.pt"

model = PPAD(STATUS, backbone_name='ViT-B/32', n_ctx=16, class_specify=False, class_token_position="end", pretrained_dir=model_path, pos_embedding=True, return_tokens=False)


model.to(device)


train_transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=Image.BICUBIC),  
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])

])


test_transform = transforms.Compose([
    transforms.Resize(size=224, interpolation=Image.BICUBIC), 
    transforms.CenterCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.39799, 0.39799, 0.39799], std=[0.32721349, 0.32721349, 0.32721349])

])

# test dataset path
test_dataset = MaskVinCXR('./Med-AD_v1_D/VinCXR', train=False, transforms=test_transform)
# test dataset path
train_dataset = MaskVinCXR('./Med-AD_v1_D/VinCXR', train=True, transforms=train_transform, shot=64, prob=0.5)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=32)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=32)


max_epoch = 100

for name, param in model.named_parameters():
    if 'ctx' in name or 'image_mask_learner' in name:

        param.requires_grad = True
    else:
        param.requires_grad = False

prompt_learner_params = model.customclip.prompt_learner.parameters()
image_mask_learner_params = model.customclip.image_mask_learner.parameters()



all_params = list(prompt_learner_params) + list(image_mask_learner_params)

optimizer = torch.optim.SGD(all_params, lr=0.001, momentum=0.9)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


lr_scheduler = scheduler_cosine
criterion = torch.nn.BCEWithLogitsLoss()


for epoch in range(max_epoch):
    model.train()
    for i, (img, labels, mask, position_name) in enumerate(trainloader):
        image = img.to(device)
        mask = mask.to(dtype=image.dtype, device=image.device)
        labels = labels.to(dtype=image.dtype, device=image.device)

        logits_per_image = model(image, mask, position_name)
        loss = criterion(logits_per_image, labels)
        optimizer.zero_grad()
        loss.backward()


        optimizer.step()
        lr_scheduler.step()
        if i % 10 == 0:
            print("epoch:", epoch, "iter", i,  "loss:", loss.item())

    model.eval()

    all_results = []
    all_labels = []
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            for i, (img, labels, masks, position_names) in enumerate(testloader):
                image = img.to(device)
                labels = labels.to(dtype=image.dtype, device=image.device)
                
                temp_results = []
                for mask, position_name in zip(masks, position_names):
                    
                    mask = mask.to(dtype=image.dtype, device=image.device)

                    logits_per_image = model(image, mask, position_name)

                    new_logits_per_image = torch.zeros((logits_per_image.shape[0],2))
                    logits_per_image = logits_per_image.cpu()

                    new_logits_per_image[:,0] = logits_per_image[:,0]
                    new_logits_per_image[:,1] = logits_per_image[:,1]
                    probs = new_logits_per_image.softmax(dim=1)
                    
                    abnormal_probs = probs[:,1]

                    temp_results.append(abnormal_probs)
                temp_results = torch.stack(temp_results, dim=0)
                max_results = temp_results.max(dim=0)[0]
                mean_results = temp_results.mean(dim=0)

                for max_result, mean_result in zip(max_results, mean_results):
                    if max_result > 0.8:
                        all_results.append(max_result)
                    else:
                        all_results.append(mean_result)

                labels = labels.argmax(dim=-1).cpu().numpy()
                all_labels.append(labels)

        
        all_results = np.array(all_results)
        all_labels = np.concatenate(all_labels)
        ap = metrics.average_precision_score(all_labels, all_results)
        auc = metrics.roc_auc_score(all_labels, all_results)
        f1 = metrics.f1_score(all_labels, all_results>0.5)
        acc = metrics.accuracy_score(all_labels, all_results>0.5)

        print("auc:", auc, "acc:", acc,  "f1:", f1, "ap:", ap )
        if epoch == max_epoch-1:
            prompt_learner_state_dict = model.customclip.prompt_learner.state_dict()
            image_mask_learner_state_dict = model.customclip.image_mask_learner.state_dict()

            current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

            # save model path
            torch.save({
                'prompt_learner_ctx': prompt_learner_state_dict,
                'image_mask_learner': image_mask_learner_state_dict,
            }, f'./epoch_{epoch}_auc_{auc}_acc_{acc}_f1_{f1}_ap_{ap}.pt')
