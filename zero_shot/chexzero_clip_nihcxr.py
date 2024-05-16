import torch
import clip
import sys
sys.path.append('..')
from PIL import Image
from myclip.datasets.zhanglab import Zhang
from myclip.datasets.chexpert import CheXpert
from myclip.datasets.rsna import Rsna
from myclip.datasets.vincxr import VinCXR
from myclip.datasets.nihcxr import NihCxr
import numpy as np
import sklearn.metrics as metrics
 

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
model_path = "/home/sunzc/medical_anomaly/MyClip/myclip/best_64_0.0001_original_35000_0.864.pt"
model.load_state_dict(torch.load(model_path, map_location=device))

STATUS = ['normal', 'pneumonia']


# STATUS = ['normal lung', 'clear lung', 'healthy lung', 'bacterial pneumonia', 'viral pneumonia', 'pleural effusion']


# text_inputs = torch.cat([clip.tokenize(f'a chest X-Ray in patients with {item}') for item in STATUS]).to(device)

text_inputs = torch.cat([clip.tokenize(f'{item}') for item in STATUS]).to(device)



# dataset = Rsna('/data/sunzc/Med-AD_v1_D/RSNA', train=False)
dataset = NihCxr('/data/sunzc/NihCxr', train=False)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

correct_num = 0
all_results = []
all_labels = []
with torch.no_grad():

    for i, (img, labels) in enumerate(trainloader):
        image = img.to(device)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        logits_per_image, logits_per_text = model(image, text_inputs)
        # break

        new_logits_per_image = torch.zeros((logits_per_image.shape[0], 2))
        logits_per_image = logits_per_image.cpu()
        # new_logits_per_image[:,0] = logits_per_image[:,0] + logits_per_image[:,1] + logits_per_image[:,2]
        # new_logits_per_image[:,1] = logits_per_image[:,3] + logits_per_image[:,4] + logits_per_image[:,5]
        new_logits_per_image[:,0] = logits_per_image[:,0]
        new_logits_per_image[:,1] = logits_per_image[:,1]
        probs = new_logits_per_image.softmax(dim=1)
        # normal_probs = probs[:,0]
        abnormal_probs = probs[:,1]
        all_results.append(abnormal_probs)
        all_labels.append(labels)



all_results = np.concatenate(all_results)
all_labels = np.concatenate(all_labels)

ap = metrics.average_precision_score(all_labels, all_results)
auc = metrics.roc_auc_score(all_labels, all_results)
f1 = metrics.f1_score(all_labels, all_results>0.5)
acc = metrics.accuracy_score(all_labels, all_results>0.5)

print("auc:", auc, "acc:", acc,  "f1:", f1, "ap:", ap )

