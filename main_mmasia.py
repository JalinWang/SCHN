import os
import time

import torch
import torch.optim as optim
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import modules as mods
import utils.evaluate as evaluate
from data.data_loader import sample_dataloader


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
               )
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
    model_cycle = mods.cycle_hash_net_3att(code_length, args.class_num).to(args.device)
    criterion_cycle = CycleLoss(code_length, args.gamma)
    optimizer_cycle = optim.Adam(model_cycle.parameters(), lr=args.lr, weight_decay=1e-5)

    retrieval_targets_onehot = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)

    B_all = torch.zeros(num_retrieval, code_length).to(args.device)

    mAP_best = 0
    mAP_best_res = None

    start = time.time()
    for it in range(args.max_iter):
        U_image_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_label_sample = torch.zeros(args.num_samples, code_length).to(args.device)
        U_latent_sample = torch.zeros(args.num_samples, code_length).to(args.device)

        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index_in_all = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset)
        sample_index_in_all = sample_index_in_all.to(args.device)
        # Create Similarity matrix
        train_targets_onehot = train_dataloader.dataset.get_onehot_targets().to(args.device)

        S_ = (train_targets_onehot @ retrieval_targets_onehot.t() > 0).float()
        S_neg1 = torch.where(S_ == 1, torch.full_like(S_, 1), torch.full_like(S_, -1))

        # Soft similarity matrix, benefit to converge
        r = S_neg1.sum() / (1 - S_neg1).sum()
        S_neg1 = S_neg1 * (1 + r) - r

        # train_data = []
        # for batch, (data, targets, batch_index_in_sample) in enumerate(train_dataloader):
        #     data, targets, batch_index_in_sample = data.to(args.device), targets.to(args.device), batch_index_in_sample.to(args.device)
        #     train_data.append([data, targets, batch_index_in_sample])

        for epoch in tqdm(range(args.max_epoch)):
            for batch, (data, targets, batch_index_in_sample) in enumerate(train_dataloader):
            # for batch_num, (data, targets, batch_index_in_sample) in enumerate(train_data):
                data, targets, batch_index_in_sample = data.to(args.device), targets.to(args.device), batch_index_in_sample.to(args.device)

                batch_sim = ((train_targets_onehot[batch_index_in_sample] @ train_targets_onehot[batch_index_in_sample].T) > 0).float()
                optimizer_cycle.zero_grad()
                output = model_cycle(data, train_targets_onehot[batch_index_in_sample], batch_sim)
                U_latent, U_image, U_label, U_image_rec, U_label_rec = output

                loss_cycle = criterion_cycle(U_latent, U_image, U_label,
                                             U_image_rec, U_label_rec,
                                             B_all[sample_index_in_all[batch_index_in_sample], :],
                                             batch_sim)
                loss_cycle.backward()
                optimizer_cycle.step()

                U_latent_sample[batch_index_in_sample, :] = U_latent.detach()
                U_image_sample[batch_index_in_sample, :], U_label_sample[batch_index_in_sample, :] = U_image.detach(), U_label.detach()

        def calc_dcc():
            B, U, S = B_all[sample_index_in_all, :], U_image_sample, S_neg1[:, sample_index_in_all]

            Q = (code_length * S).t() @ U + args.gamma * U

            for bit in range(code_length):
                q = Q[:, bit]
                u = U[:, bit]
                B_prime = torch.cat((B[:, :bit], B[:, bit + 1:]), dim=1)
                U_prime = torch.cat((U[:, :bit], U[:, bit + 1:]), dim=1)

                B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t() + args.alpha * U_label_sample[:, bit] + args.beta*U_latent_sample[:, bit]).sign()
            return B

        B_all[sample_index_in_all, :] = calc_dcc()

        logger.debug('[iter:{}/{}][iter_time:{:.2f}]'.format(it + 1, args.max_iter, time.time() - iter_start))

        if (it + 1) % args.eval_iter == 0 or it + 1 == args.max_iter:
            # del train_data
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

