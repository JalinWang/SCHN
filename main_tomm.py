import torch
import torchvision.models
import torch.optim as optim
import os
import time

from torch import Tensor
from tqdm import tqdm

import utils.evaluate as evaluate

from loguru import logger
from data.data_loader import sample_dataloader

import modules as mods


def CycleLoss(code_length, gamma):
    def forward(U_latent, U_image, U_label, U_image_rec, U_label_rec, B, S):
        def feature_loss(F, B, S):
            hash_loss = ((code_length * S - F @ B.t()) ** 2).mean() * 12 / code_length
            quantization_loss = ((F - B) ** 2).mean()
            return hash_loss + gamma * quantization_loss

        loss = (
                       feature_loss(U_image, B, S)
                       + feature_loss(U_label, B, S)
                       + feature_loss(U_latent, B, S)
                       + ((U_image_rec - U_image) ** 2).mean() + ((U_label_rec - U_label) ** 2).mean()
                       # + ((U_image_rec - U_image.detach()) ** 2).sum() + ((U_label_rec - U_label.detach()) ** 2).sum()
               )
        # / (U_image.shape[0] * B.shape[0])

        return loss

    return forward


def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        logdir,
        args
):
    num_retrieval = len(retrieval_dataloader.dataset)

    # Initialization
    model_cycle = cycle_hash_net_3att(code_length, args.class_num).to(args.device)
    criterion_cycle = CycleLoss(code_length, args.gamma)
    optimizer_cycle = optim.Adam(model_cycle.parameters(), lr=args.lr, weight_decay=1e-5)


    model_D = ReconstructionDiscriminatorNetwork(code_length).to(args.device)
    criterion_D = nn.BCELoss()
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_G = optim.Adam(model_cycle.model_cycle.parameters(), lr=args.lr, weight_decay=1e-5)


    def adjusting_learning_rate(optimizer, iter):
        if args.max_iter == 50:
            update_list = [5, 40]
        else:
            update_list = [130]
        if iter in update_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 5

    # def initialize_weights(net):
    #     for m in net.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
    #             m.bias.data.zero_()
    #         # elif isinstance(m, nn.ConvTranspose2d):
    #         #     m.weight.data.normal_(0, 0.02)
    #         #     m.bias.data.zero_()
    #         # elif isinstance(m, nn.Linear):
    #         #     # nn.init.kaiming_normal (m.weight.data, mode='fan_in')
    #         #     # nn.init.xavier_normal(m.weight.data)
    #         #     # nn.init.normal(m.weight.data, 0, 0.01)
    #         #     m.weight.data.normal_(0, 0.01)
    #         #     # nn.init.uniform (m.weight.data, a=-0.1, b=0.1)
    #         #     m.bias.data.zero_()
    # # initialize_weights(model_cycle)
    # # initialize_weights(model_D)

    retrieval_targets_onehot = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)

    B_all = torch.zeros(num_retrieval, code_length).to(args.device)

    real_label = 1.
    fake_label = 0.


    mAP_best = 0
    mAP_best_res = None

    start = time.time()
    for it in range(args.max_iter):
        U_image_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_label_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_latent_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        # B_samples = torch.zeros(args.num_samples, code_length).to(args.device)

        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index_in_all = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset)
        sample_index_in_all = sample_index_in_all.to(args.device)
        # Create Similarity matrix
        train_targets_onehot = train_dataloader.dataset.get_onehot_targets().to(args.device)

        S_ = (train_targets_onehot @ retrieval_targets_onehot.t() > 0).float()
        S_neg1 = torch.where(S_ == 1, torch.full_like(S_, 1), torch.full_like(S_, -1))

        # S_01 = (S_ == 1).float()
        # Z = torch.true_divide(S_01, S_01.sum(1).view([-1, 1]))

        # Soft similarity matrix, benefit to converge
        r = S_neg1.sum() / (1 - S_neg1).sum()
        S_neg1 = S_neg1 * (1 + r) - r

        train_data = []
        for batch, (data, targets, batch_index_in_sample) in enumerate(train_dataloader):
            data, targets, batch_index_in_sample = data.to(args.device), targets.to(args.device), batch_index_in_sample.to(args.device)
            train_data.append([data, targets, batch_index_in_sample])

        for epoch in tqdm(range(args.max_epoch)):
            # for batch, (data, targets, batch_index_in_sample) in enumerate(train_dataloader):
            for batch_num, (data, targets, batch_index_in_sample) in enumerate(train_data):
                data, targets, batch_index_in_sample = data.to(args.device), targets.to(args.device), batch_index_in_sample.to(args.device)

                # actual_batch_size = data.shape[0]
                # u_ind = torch.linspace(epoch * batch_num,
                #                        min(args.num_samples, (batch_num+1) * args.batch_size) - 1,
                #                        actual_batch_size, dtype=torch.long)

                batch_sim = ((train_targets_onehot[batch_index_in_sample] @ train_targets_onehot[batch_index_in_sample].T) > 0).float()

                optimizer_cycle.zero_grad()
                output = model_cycle(data, train_targets_onehot[batch_index_in_sample], batch_sim)
                U_latent, U_image, U_label, U_image_rec, U_label_rec = output

                loss_cycle = criterion_cycle(U_latent, U_image, U_label,
                                             U_image_rec, U_label_rec,
                                             B_all[sample_index_in_all[batch_index_in_sample], :],
                                             batch_sim)
                # with torch.autograd.set_detect_anomaly(True):
                # loss_cycle.backward(retain_graph=True)

                loss_cycle.backward()
                optimizer_cycle.step()


                output = model_cycle(data, train_targets_onehot[batch_index_in_sample], batch_sim)
                U_latent, U_image, U_label, U_image_rec, U_label_rec = output



                optimizer_D.zero_grad()

                outputs = torch.cat(model_D(U_image.detach(), U_label.detach())).view(-1)
                label_criterion = torch.full_like(outputs, real_label, dtype=torch.float, device=args.device)
                lossD_real = criterion_D(outputs, label_criterion)

                outputs = torch.cat(model_D(U_image_rec.detach(), U_label_rec.detach())).view(-1)
                label_criterion = torch.full_like(outputs, fake_label, dtype=torch.float, device=args.device)
                lossD_fake = criterion_D(outputs, label_criterion)
                
                lossD = lossD_real + lossD_fake
                lossD.backward()
                optimizer_D.step()


                optimizer_G.zero_grad()
                # output = model_cycle(data, train_targets_onehot[batch_index_in_sample], batch_sim)
                # *_, U_image_rec, U_label_rec = output
                outputs = torch.cat(model_D(U_image_rec, U_label_rec)).view(-1)
                label_criterion = torch.full_like(outputs, real_label, dtype=torch.float, device=args.device)
                lossG = criterion_D(outputs, label_criterion)
                lossG.backward()
                optimizer_G.step()
      


                U_latent_sample[batch_index_in_sample, :] = U_latent.detach()
                U_image_sample[batch_index_in_sample, :], U_label_sample[batch_index_in_sample, :] = U_image.detach(), U_label.detach()

        adjusting_learning_rate(optimizer_G, it)
        adjusting_learning_rate(optimizer_D, it)

        def calc_dcc():
            B, U, S = B_all[sample_index_in_all, :], U_image_sample, S_neg1[:, sample_index_in_all]

            Q = (code_length * S).t() @ U + args.gamma * U

            # expand_U[sample_index_in_all, :] = (2 * U_label + U_gcn)
            # # expand_U[sample_index_in_all, :] =  U_image +5 *  U_label
            for bit in range(code_length):
                q = Q[:, bit]
                u = U[:, bit]
                B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
                U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

                B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t() + args.alpha * U_label_sample[:, bit] + args.beta*U_latent_sample[:, bit]).sign()
                # B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t() + 2 * U_label_sample[:, bit]).sign()
            return B

        # if it + 1 <= args.max_iter - 1:
        B_all[sample_index_in_all, :] = calc_dcc()

        logger.debug('[iter:{}/{}][iter_time:{:.2f}]'.format(it + 1, args.max_iter, time.time() - iter_start))

        if (it + 1) % args.eval_iter == 0 or it + 1 == args.max_iter:
            del train_data
            # Evaluate
            query_code = generate_code(model_cycle.model_image, query_dataloader, code_length, args.device)
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B_all,
                # B_samples,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets_onehot.to(args.device),
                # train_targets_onehot[sample_index_in_all],
                args.device,
                args.topk,
            )
            logger.info(
                "[Evaluation][dataset:{}][bits:{}][iter:{}/{}][mAP:{:.4f}]".format(args.dataset, code_length, it + 1,
                                                                                   args.max_iter, mAP))
            if mAP > mAP_best:
                mAP_best = mAP

                mAP_best_res = [query_code.cpu(), B_all.cpu(), query_dataloader.dataset.get_onehot_targets().cpu(), retrieval_targets_onehot.cpu()]
            #     # Save checkpoints
            #     gen_name = lambda name: os.path.join(logdir, f'{args.dataset}-{code_length}bits-{mAP}-{name}.t')
            #     torch.save(query_code.cpu(), gen_name("query_code"))
            #     torch.save(B_all.cpu(), gen_name("database_code"))
            #     torch.save(query_dataloader.dataset.get_onehot_targets().cpu(), gen_name("query_targets"))
            #     torch.save(retrieval_targets_onehot.cpu(), gen_name("database_targets"))

    logger.info('[Training time:{:.2f}]'.format(time.time() - start))

    if mAP_best_res is not None:
        query_code, database_code, query_targets, database_targets = mAP_best_res

        # Save checkpoints
        gen_name = lambda name: os.path.join(logdir, f'{args.dataset}-{code_length}bits-{mAP_best}-{name}.t')
        torch.save(query_code, gen_name("query_code"))
        torch.save(database_code, gen_name("database_code"))
        torch.save(query_targets, gen_name("query_targets"))
        torch.save(database_targets, gen_name("database_targets"))
        # torch.save(model_image, gen_name("model_image"))

    return mAP_best


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            _, hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code


from torch import nn
import torch
from torch.nn import functional as F

from torch.hub import load_state_dict_from_url

hidden_size_image = 4096
hidden_size_label = 512
# hidden_size_latent = hidden_size_image + hidden_size_label
# hidden_size_latent = hidden_size_label
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

        self.gc1 = mods.GraphConvolution(nfeat, nhid)
        # self.gc2 = mods.GraphConvolution(nhid, nclass)
        self.hash_code_layer = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        feature = F.relu(self.gc1(x, adj))
        x = feature
        x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.sigmoid(self.gc2(x, adj))
        x = torch.sigmoid(self.hash_code_layer(x))
        return feature, x

class CycleNet_lat_as_key(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.model_image_rec = MLP_feature_output(hidden_size_latent, code_length)
        self.model_label_rec = MLP_feature_output(hidden_size_latent, code_length)
        # self.model_image_rec = MLP_feature_output(hidden_size_label, code_length)
        # self.model_label_rec = MLP_feature_output(hidden_size_image, code_length)

        self.gcn = GCN_feature_output(hidden_size_label, hidden_size_latent, code_length, 0.5)


        self.self_attention_lat = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_image)
        self.self_attention_image_rec = nn.MultiheadAttention(hidden_size_latent, 1, kdim=hidden_size_latent, vdim=hidden_size_label)
        self.self_attention_label_rec = nn.MultiheadAttention(hidden_size_latent, 1, kdim=hidden_size_latent, vdim=hidden_size_image)
        # self.self_attention = nn.MultiheadAttention(hidden_size_label, 1, vdim=hidden_size_image)
        # self.self_attention_image_rec = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_latent)
        # self.self_attention_label_rec = nn.MultiheadAttention(hidden_size_image, 1, kdim=hidden_size_image, vdim=hidden_size_latent)

    def forward(self, F_image, F_label, S=None):
        # F_image, F_label = F_image.detach().unsqueeze(1), F_label.detach().unsqueeze(1)
        F_image, F_label = F_image.unsqueeze(1), F_label.unsqueeze(1)

        fused_for_latent, weight = self.self_attention_lat(F_label, F_label, F_image)

        latent_space, U_latent = self.gcn(fused_for_latent.squeeze(1), S)
        latent_space_ = latent_space.unsqueeze(1)

        # fused_for_image, weight = self.self_attention_image_rec(F_label, F_label, latent_space_)
        # fused_for_label, weight = self.self_attention_label_rec(F_image, F_image, latent_space_)
        fused_for_image, weight = self.self_attention_image_rec(latent_space_, latent_space_, F_label)
        fused_for_label, weight = self.self_attention_label_rec(latent_space_, latent_space_, F_image)

        _, U_image_rec = self.model_image_rec(fused_for_image.squeeze(1))
        _, U_label_rec = self.model_label_rec(fused_for_label.squeeze(1))

        return U_latent, latent_space, U_image_rec, U_label_rec

class CycleNet_twice(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.model_image_rec = MLP_feature_output(hidden_size_label, code_length)
        self.model_label_rec = MLP_feature_output(hidden_size_image, code_length)

        self.gcn = GCN_feature_output(hidden_size_label, hidden_size_latent, code_length, 0.5)

        self.self_attention_lat = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_image)
        self.self_attention_image_rec = nn.MultiheadAttention(hidden_size_label, 1, kdim=hidden_size_label, vdim=hidden_size_latent)
        self.self_attention_label_rec = nn.MultiheadAttention(hidden_size_image, 1, kdim=hidden_size_image, vdim=hidden_size_latent)

    def forward(self, F_image, F_label, S=None):
        # F_image, F_label = F_image.detach().unsqueeze(1), F_label.detach().unsqueeze(1)
        F_image, F_label = F_image.unsqueeze(1), F_label.unsqueeze(1)

        fused_for_latent, weight = self.self_attention_lat(F_label, F_label, F_image)

        latent_space, U_latent = self.gcn(fused_for_latent.squeeze(1), S)
        latent_space_ = latent_space.unsqueeze(1)

        fused_for_image, weight = self.self_attention_image_rec(F_label, F_label, latent_space_)
        fused_for_label, weight = self.self_attention_label_rec(F_image, F_image, latent_space_)

        _, U_image_rec = self.model_image_rec(fused_for_image.squeeze(1))
        _, U_label_rec = self.model_label_rec(fused_for_label.squeeze(1))

        return U_latent, latent_space, U_image_rec, U_label_rec

class CycleNet(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.model_image_rec = MLP_feature_output(hidden_size_label, code_length)
        self.model_label_rec = MLP_feature_output(hidden_size_image, code_length)

        self.gcn = GCN_feature_output(hidden_size_label, hidden_size_latent, code_length, 0.5)

        # self.self_attention = nn.MultiheadAttention(hidden_size_label, 1, vdim=hidden_size_image)
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
        self.model_cycle = CycleNet_lat_as_key(code_length)

    def forward(self, images, labels, S=None):
        F_image, U_image = self.model_image(images)
        F_label, U_label = self.model_label(labels)
        # U_latent, _, U_image_rec, U_label_rec = self.model_cycle(F_image, F_label, S)
        U_latent, _, U_image_rec, U_label_rec = self.model_cycle(F_image.detach(), F_label.detach(), S)

        return U_latent, U_image, U_label, U_image_rec, U_label_rec


def cycle_hash_net_3att(*args):
    model = CycleHashNet(*args)

    return model


class ReconstructionDiscriminatorNetwork(nn.Module):
    def __init__(self, code_length):
        super().__init__()

        self.model_imageD = MLP_feature_output(code_length, 1)
        self.model_labelD = MLP_feature_output(code_length, 1)

    def forward(self, U_image_rec, U_label_rec):
        _, output_i = self.model_imageD(U_image_rec)
        _, output_l = self.model_labelD(U_label_rec)

        # turn outputs of TanH() into probabilities
        ret = [ (output_i+1)/2, (output_l+1)/2 ]

        return ret