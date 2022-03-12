
# For the convenience of tracking changes in network architectures, I put the all codes here, which makes the file a little bit ugly.
import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from .gcn.models import GCN, GraphConvolution

hidden_size_image = 4096
hidden_size_label = 512
hidden_size_latent = 1024


def fuse(a, b):
    return torch.cat([a, b], 1)


class AlexNet_feature_output(nn.Module):
    def __init__(self, code_length):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        feature = self.features(x)
        x = feature
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        hash_code = self.hash_layer(x)
        return x, hash_code


def alexnet_feature_output(*args):
    model = AlexNet_feature_output(*args)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    model.load_state_dict(state_dict, strict=False)

    return model


class MLP_feature_output(nn.Module):
    def __init__(self, class_num, code_length):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(class_num, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            # nn.Linear(512, code_length),
            # nn.Tanh()
        )
        self.hash_layer = nn.Sequential(
            nn.Linear(512, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        hash_code = self.hash_layer(x)
        return x, hash_code


class GCN_feature_output(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()

        self.gc = GraphConvolution(nfeat, nhid)
        self.hash_code_layer = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        feature = F.relu(self.gc(x, adj))
        x = feature
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.sigmoid(self.hash_code_layer(x))
        return feature, x

class CycleNet(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.model_image_rec = MLP_feature_output(hidden_size_latent, code_length)
        self.model_label_rec = MLP_feature_output(hidden_size_latent, code_length)

        self.gcn = GCN_feature_output(hidden_size_label, hidden_size_latent, code_length, 0.5)


        self.self_attention_lat = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_image)
        self.self_attention_image_rec = nn.MultiheadAttention(hidden_size_latent, 1, kdim=hidden_size_latent, vdim=hidden_size_label)
        self.self_attention_label_rec = nn.MultiheadAttention(hidden_size_latent, 1, kdim=hidden_size_latent, vdim=hidden_size_image)


    def forward(self, F_image, F_label, S=None):
        F_image, F_label = F_image.unsqueeze(1), F_label.unsqueeze(1)

        fused_for_latent, weight = self.self_attention_lat(F_label, F_label, F_image)

        latent_space, U_latent = self.gcn(fused_for_latent.squeeze(1), S)
        latent_space_ = latent_space.unsqueeze(1)

        fused_for_image, weight = self.self_attention_image_rec(latent_space_, latent_space_, F_label)
        fused_for_label, weight = self.self_attention_label_rec(latent_space_, latent_space_, F_image)

        _, U_image_rec = self.model_image_rec(fused_for_image.squeeze(1))
        _, U_label_rec = self.model_label_rec(fused_for_label.squeeze(1))

        return U_latent, latent_space, U_image_rec, U_label_rec

    def __init__(self, code_length):
        super().__init__()

        self.model_image_rec = MLP_feature_output(hidden_size_label, code_length)
        self.model_label_rec = MLP_feature_output(hidden_size_image, code_length)

        self.gcn = GCN_feature_output(hidden_size_label, hidden_size_latent, code_length, 0.5)

        self.self_attention_lat = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_image)
        self.self_attention_image_rec = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_latent)
        self.self_attention_label_rec = nn.MultiheadAttention(hidden_size_image, 1, kdim=hidden_size_image, vdim=hidden_size_latent)

    def forward(self, F_image, F_label, S=None):
        F_image, F_label = F_image.detach().unsqueeze(1), F_label.detach().unsqueeze(1)

        fused_for_latent, weight = self.self_attention_lat(F_label, F_label, F_image)

        latent_space, U_latent = self.gcn(fused_for_latent.squeeze(1), S)
        latent_space_ = latent_space.unsqueeze(1)

        fused_for_image, weight = self.self_attention_image_rec(F_label, F_label, latent_space_)
        fused_for_label, weight = self.self_attention_label_rec(F_image, F_image, latent_space_)

        _, U_image_rec = self.model_image_rec(fused_for_image.squeeze(1))
        _, U_label_rec = self.model_label_rec(fused_for_label.squeeze(1))

        return U_latent, latent_space, U_image_rec, U_label_rec

class CycleHashNet(nn.Module):
    def __init__(self, code_length, class_num, ):
        super().__init__()

        self.model_image = alexnet_feature_output(code_length)
        self.model_label = MLP_feature_output(class_num, code_length)
        self.model_cycle = CycleNet(code_length)

    def forward(self, images, labels, S=None):
        F_image, U_image = self.model_image(images)
        F_label, U_label = self.model_label(labels)
        U_latent, _, U_image_rec, U_label_rec = self.model_cycle(F_image, F_label, S)

        return U_latent, U_image, U_label, U_image_rec, U_label_rec


def cycle_hash_net_3att(*args):
    model = CycleHashNet(*args)

    return model


