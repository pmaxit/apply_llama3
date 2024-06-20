import torch
from transformers import TrainerCallback
from torchvision.transforms import Lambda

class MixupCallback(TrainerCallback):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def on_train_batch_start(self, args, state, control, **kwargs):
        # Access the batch data and labels
        batch = kwargs['batch']
        inputs, labels = batch['pixel_values'], batch['labels']

        # apply mixup transformation
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        index = torch.randperm(inputs.size(0)).to(inputs.device)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]

        # Update the batch with mixed data and labels
        kwargs["batch"] = {"pixel_values": mixed_inputs, "labels": mixed_labels}


class CutMixCallback(TrainerCallback):
    def __init__(self, alpha = 1.0,labelsmoothing=False):
        self.alpha = alpha
        self.labelsmoothing=labelsmoothing

    def on_train_batch_start(self, args, state, control, **kwargs):
        batch = kwargs['batch']
        inputs, labels = batch['pixel_values'], batch['labels']

        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample()
        rand_index = torch.randperm(inputs.size()[0]).to(inputs.device)
        label_a = labels
        label_b = labels[rand_index]

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(inputs.size(), lam)

        # Cut and mix images
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, bby1:bby2, bbx1:bbx2] = inputs[rand_index][:, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda to match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

        if self.labelsmoothing:
            lam = 0.9*lam + 0.1
        # Cut and mix labels
        mixed_labels = lam * labels + (1 - lam) * labels[rand_index]

        kwargs["batch"] = {"pixel_values": mixed_inputs, "labels": mixed_labels}


    def _rand_bbox(self, size, lam):
        W = size[-1]
        H = size[-2]
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.round(W * cut_rat).type(torch.long)
        cut_h = torch.round(H * cut_rat).type(torch.long)

        # Uniform
        cx = torch.randint(W, (1,))[0]
        cy = torch.randint(H, (1,))[0]

        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class LabelSmoothingCallback(TrainerCallback):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def on_train_batch_start(self, args, state, control, **kwargs):
        batch = kwargs["batch"]
        labels = batch["labels"]  
        
        # Apply label smoothing if labels are not already smoothed
        if labels.dtype != torch.float:
            num_classes = labels.max().item() + 1  # Assuming class indices start from 0
            smoothed_labels = (1 - self.epsilon) * F.one_hot(labels, num_classes=num_classes) + self.epsilon / num_classes
            kwargs["batch"]["labels"] = smoothed_labels


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def freeze(model, freeze=True):
    for name, param in model.model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = not freeze

    print('parameters ',count_parameters(model))