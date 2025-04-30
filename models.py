import torch
import torch.nn as nn
import torch.nn.functional as F

# Evidence activation functions
def relu_evidence(x): return F.relu(x)
def exp_evidence(x): return torch.exp(x)
def softplus_evidence(x): return F.softplus(x)

class BasicResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes=10, edl=False, evidence_fn="relu"):
        super().__init__()
        self.edl = edl
        ev_map = {"relu": relu_evidence, "exp": exp_evidence, "softplus": softplus_evidence}
        self.evidence_fn = ev_map[evidence_fn]

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.layer2 = BasicResBlock(16, 32, 2)
        self.layer3 = BasicResBlock(32, 64, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).view(out.size(0), -1)
        out = self.fc(out)
        if self.edl:
            evidence = self.evidence_fn(out)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            belief = evidence / (S - alpha.size(1))
            uncertainty = alpha.size(1) / S
            prob = alpha / S
            return {
                'evidence': evidence,
                'alpha': alpha,
                'belief': belief,
                'uncertainty': uncertainty,
                'probability': prob
            }
        else:
            return F.softmax(out, dim=1)
        
class BasicCBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        # self.batchNorm = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return self.relu(self.batchNorm(self.conv(x)))
        return self.relu(self.conv(x))

class LeNet(nn.Module):
    def __init__(self, num_classes=10, edl=False, evidence_fn="relu"):
        super().__init__()
        self.edl = edl
        ev_map = {"relu": relu_evidence, "exp": exp_evidence, "softplus": softplus_evidence}
        self.evidence_fn = ev_map[evidence_fn]
        self.layer1 = BasicCBR(1,6,5,1,0)
        self.layer2 = BasicCBR(6,16,5,1,0)
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.maxpool(out)
        out = self.layer2(out)
        out = self.maxpool(out)
        
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        if self.edl:
            evidence = self.evidence_fn(out)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            belief = evidence / (S - alpha.size(1))
            uncertainty = alpha.size(1) / S
            prob = alpha / S
            return {
                'evidence': evidence,
                'alpha': alpha,
                'belief': belief,
                'uncertainty': uncertainty,
                'probability': prob
            }
        else:
            return F.softmax(out, dim=1)


def get_model(config):
    model_name = config['model']['type']
    num_classes = config['model']['num_classes']
    edl = config['edl']['enabled']
    evidence_fn = config['edl']['evidence_func']

    if model_name == 'resnet':
        return ResNet(num_classes=num_classes, edl=edl, evidence_fn=evidence_fn)
    elif model_name == 'lenet':
        return LeNet(num_classes=num_classes, edl=edl, evidence_fn=evidence_fn)
    else:
        raise ValueError(f"Unknown model type: {model_name}")