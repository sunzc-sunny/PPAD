import torch
import clip
import sys
sys.path.append('..')
from PIL import Image
from ppad_clip.datasets.chexpert import CheXpert
import numpy as np
import sklearn.metrics as metrics


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)
# pre-trained model path
model_path = ""
model.load_state_dict(torch.load(model_path, map_location=device))


STATUS = ['normal.', 'pneumonia.']
text_inputs = torch.cat([clip.tokenize(f'{item}') for item in STATUS]).to(device)


# test dataset path
dataset = CheXpert('', train=False)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=0)

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

        new_logits_per_image = torch.zeros((logits_per_image.shape[0],2))
        logits_per_image = logits_per_image.cpu()
        new_logits_per_image[:,0] = logits_per_image[:,0]
        new_logits_per_image[:,1] = logits_per_image[:,1]
        probs = new_logits_per_image.softmax(dim=1)
        abnormal_probs = probs[:,1]
        all_results.append(abnormal_probs)
        all_labels.append(labels)



all_results = np.concatenate(all_results)
all_labels = np.concatenate(all_labels)

ap = metrics.average_precision_score(all_labels, all_results)
auc = metrics.roc_auc_score(all_labels, all_results)
f1 = metrics.f1_score(all_labels, all_results>0.5)
acc = metrics.accuracy_score(all_labels, all_results>0.5)

print("acc:", acc, "auc:", auc, "f1:", f1, "ap:", ap )

