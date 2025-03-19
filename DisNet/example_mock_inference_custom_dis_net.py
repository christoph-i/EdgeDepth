import random

import torch

from distance_estimation_project.models.disnet import DisNet
from distance_estimation_project.trainers.trainer_mlp import TrainerMLP


from train_custom_dis_net import Args

MODEL_PATH = r".../.../best.pth"

DEVICE = "cpu"




model_args = Args()
model = DisNet(model_args)

model.load_w(MODEL_PATH)

#optimizer = model.get_optimizer()

# checkpoint = torch.load(MODEL_PATH)
# model.load_state_dict(checkpoint['state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer'])



model.eval()

correct = [random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), 0.88, 1.75, 0.55, 0.30]
correct[2] = correct[0] + correct[1]
different = [0.5, 0.5, 0.5, 1.75, 0.55, 0.30]
values = [correct, correct, different]


with torch.no_grad():
    for value in values:
        x = torch.tensor(value).type(torch.float32).to(DEVICE)
        dist_pred = model(x)
        print(f"Input = {value}, dist_pred = {dist_pred}")