import torch
import clip
import sys
sys.path.append('..')
from PIL import Image
from ppad_clip.datasets.zhanglab import Zhang
import numpy as np
import sklearn.metrics as metrics
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

# pre-trained model path
model_path = ""

model.load_state_dict(torch.load(model_path, map_location=device))

STATUS = ['normal.', 'pneumonia.']
text_inputs = torch.cat([clip.tokenize(f'{item}') for item in STATUS]).to(device)

# test dataset path
dataset = Zhang('', train=False)

testloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)


correct_num = 0
all_results = []
all_labels = []

# test normal image path
normal_items = os.listdir(os.path.join('', 'normal_256'))
# test pneumonia image path
pneumonia_items = os.listdir(os.path.join('', 'pneumonia_256'))
items = normal_items + pneumonia_items



with torch.no_grad():

    for i, (img, labels) in enumerate(testloader):
        image = img.to(device)

        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        logits_per_image, logits_per_text = model(image, text_inputs)

        new_logits_per_image = torch.zeros((logits_per_image.shape[0],2))
        logits_per_image = logits_per_image.cpu()
        new_logits_per_image[:,0] = logits_per_image[:,0]
        new_logits_per_image[:,1] = logits_per_image[:,1]
        probs = new_logits_per_image.softmax(dim=1)
        abnormal_probs = probs[:,1]
        all_results.append(abnormal_probs)
        labels = labels.argmax(dim=-1).cpu().numpy()
        all_labels.append(labels)



all_results = np.concatenate(all_results)
all_labels = np.concatenate(all_labels)
ap = metrics.average_precision_score(all_labels, all_results)
auc = metrics.roc_auc_score(all_labels, all_results)
f1 = metrics.f1_score(all_labels, all_results>0.5)
acc = metrics.accuracy_score(all_labels, all_results>0.5)


is_correct = (all_results>0.5) == all_labels


print("acc:", acc, "auc:", auc,  "f1:", f1, "ap:", ap )

