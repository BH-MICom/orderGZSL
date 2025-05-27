from collections import Counter
from math import e
import traceback
import pandas as pd
import torch
from utils.utils import *
from core.evaluate import Accuracy, WeightedAccuracy
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import kl_divergence
from torch.distributions import Normal
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def accumulate_features(features, labels, ordinal):
    """
    根据同一类数据的所有 features 计算 mean 和 covariance
    [clb] 假设输入为(8,128),有三个类别，则最后的输出为(3,128),即对于某个label的所有样本，计算均值和方差
    ordinal: [1, 0, 2]
    """
    means, vars = [], []

    for label in ordinal:
        means.append(features[labels == label, :].mean(dim=0).type(torch.cuda.FloatTensor))
        vars.append(features[labels == label, :].var(dim=0, unbiased=False).type(torch.cuda.FloatTensor))

    return means, vars


def get_joint_priors(mus, nablas, sigmas, num_classes):
    # 得到先验分布的 mu 和 covariance matrix
    # 得到上三角部分为 1 的矩阵
    # 通过矩阵相乘来得到 joint distribution 中的 mu 和 sigma
    # 将 nablas 和 sigmas 转换为有效的值
    # [clb] 将nalbas转化为正值
    nablas = torch.exp(nablas)
    # [clb] 将sigmas缩放到（0，1）同时保持方差与类别间偏移量的比例关系
    sigmas = (torch.sigmoid(sigmas) * nablas / 3) ** 2
    # [clb] 最终triu是一个下三角矩阵
    triu = torch.triu(torch.ones(num_classes, num_classes)).T.cuda()
    # [clb] 假设nablas表示为[[2*n1],[n2],[n3]],操作后为[[n1],[n1+n2],[n1+n2+n3]]
    joint_nablas = torch.matmul(triu, nablas)  # (K, d)
    # [clb] 得到论文中的均值分布
    joint_mus = mus + joint_nablas  # (K, d)
    # [clb] 循环之后的mat:
    '''
        tensor([[[1., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.]],

                [[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 0.]],

                [[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]]])
    '''
    mat = torch.zeros(num_classes, num_classes, num_classes).cuda()
    for i in range(num_classes):
        for j in range(num_classes):
            mat[i, j][:min(i, j) + 1] = 1

    # [clb] 得到论文中的协方差分布
    joint_sigmas = torch.matmul(mat, sigmas)

    return joint_mus.type(torch.cuda.FloatTensor), joint_sigmas.type(torch.cuda.FloatTensor), nablas.type(torch.cuda.FloatTensor)

def negative_log_likelihood(features, targets, joint_mus, joint_sigmas, num_classes, neg_weight=0.1):
    log_likelihood = 0
    num_samples = features.size(0)
    
    for i in range(num_samples):  # 遍历每个样本
        # 获取该样本类别标签
        label = targets[i]
        
        # 计算正确类别的负对数似然
        mu = joint_mus[label]  # shape: (32,)
        sigma = torch.diag(joint_sigmas[label, label])  # shape: (32, 32)
        prior = MultivariateNormal(mu, covariance_matrix=sigma)
        log_likelihood += prior.log_prob(features[i])  # 正类别对数似然
        
        # 计算错误类别的负对数似然并赋予负权重
        # for j in range(num_classes):
        #     if j != label:  # 跳过正确类别
        #         mu_j = joint_mus[j]
        #         sigma_j = torch.diag(joint_sigmas[j, j])
        #         prior_j = MultivariateNormal(mu_j, covariance_matrix=sigma_j)
        #         log_likelihood -= neg_weight * prior_j.log_prob(features[i])  # 对其他类别赋予负权重
    
    # 返回负对数似然
    return -log_likelihood / num_samples

def kl_multi_loss(features, targets, joint_mus, joint_sigmas):
    """
    计算KL散度损失：对于每个样本，最小化与正确类别的KL散度。
    neg_weight 用于控制非正确类别的KL散度对总损失的影响。
    """
    kl_loss = 0
    num_samples = features.size(0)
    
    for i in range(num_samples):  # 遍历每个样本
        # 获取样本对应的类别标签
        label = targets[i]
        
        # 计算正确类别的KL散度（最小化）
        mu = joint_mus[label]  
        sigma = torch.diag(joint_sigmas[label, label])  
        post = MultivariateNormal(features[i], covariance_matrix=torch.eye(features.size(1)).cuda())  # 假设特征协方差为单位矩阵
        prior = MultivariateNormal(mu, covariance_matrix=sigma)
        
        # 最小化与正确类别的KL散度
        kl_loss += torch.distributions.kl.kl_divergence(post, prior)
        
    return kl_loss / num_samples  # 返回平均损失

def kl_multi_contrastive_loss(features, targets, joint_mus, joint_sigmas, num_classes, neg_weight=0.1):
    """
    计算KL散度对比损失：对于每个样本，最小化与正确类别的KL散度，最大化与其他类别的KL散度。
    neg_weight 用于控制非正确类别的KL散度对总损失的影响。
    """
    kl_loss = 0
    num_samples = features.size(0)
    
    for i in range(num_samples):  # 遍历每个样本
        # 获取样本对应的类别标签
        label = targets[i]
        
        # 计算正确类别的KL散度（最小化）
        mu = joint_mus[label]  
        sigma = torch.diag(joint_sigmas[label, label])  
        post = MultivariateNormal(features[i], covariance_matrix=torch.eye(features.size(1)).cuda())  # 假设特征协方差为单位矩阵
        prior = MultivariateNormal(mu, covariance_matrix=sigma)
        
        # 最小化与正确类别的KL散度
        kl_loss += torch.distributions.kl.kl_divergence(post, prior)
        
        # # 计算错误类别的KL散度（最大化）
        for j in range(num_classes):
            if j != label:  # 跳过正确类别
                mu_j = joint_mus[j]
                sigma_j = torch.diag(joint_sigmas[j, j])
                prior_j = MultivariateNormal(mu_j, covariance_matrix=sigma_j)
                
                # 最大化与其他类别的KL散度，实际上是最小化负的KL散度
                kl_loss -= neg_weight * torch.distributions.kl.kl_divergence(post, prior_j)

    return kl_loss / num_samples  # 返回平均损失



def kl_independent_loss(features, targets, joint_mus, joint_sigmas, seen_classes, dataset_ordinal, neg_weight=0.1):
    """
    计算KL散度损失：
    - 最小化特征与先验类别的KL散度
    - 最大化特征与非真实类别的KL散度
    """
    kl_loss = 0
    means, vars = accumulate_features(features, targets, dataset_ordinal)  # 计算各类的均值和方差
    num_samples = features.size(0)
    device = features.device

    for d in range(features.size(1)):  # 针对每个维度独立考虑
        try:
            for i in seen_classes:
                post = Normal(means[i][d], vars[i][d].sqrt())
                prior = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
                kl_loss += kl_divergence(post, prior).mean()

            # # 对比损失：计算不同类别之间的KL散度差异
            # kl_matrix = compute_kl_matrix(means, vars, joint_mus, joint_sigmas)

            # for i in range(len(dataset_ordinal)):
            #     for j in range(len(dataset_ordinal)):
            #         if i != j:
            #             # 最大化非真实类别的 KL 散度
            #             kl_loss -= neg_weight * abs(j - i) * kl_matrix[i, j]

        except Exception as e:
            print(f"Error calculating KL divergence: {e}")
            continue
    kl_loss = kl_loss / features.size(1)

    return kl_loss / num_samples


def kl_independent_contrastive_loss(features, targets, joint_mus, joint_sigmas, seen_classes, dataset_ordinal, neg_weight=0.1):
    """
    计算KL散度损失：
    - 最小化特征与先验类别的KL散度
    - 最大化特征与非真实类别的KL散度
    """
    kl_loss = 0
    means, vars = accumulate_features(features, targets, dataset_ordinal)  # 计算各类的均值和方差
    num_samples = features.size(0)
    device = features.device

    for d in range(features.size(1)):  # 针对每个维度独立考虑
        try:
            for i in seen_classes:
                post = Normal(means[i][d], vars[i][d].sqrt())
                prior = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
                kl_loss += kl_divergence(post, prior).mean()

            # # 对比损失：计算不同类别之间的KL散度差异
            # kl_matrix = compute_kl_matrix(means, vars, joint_mus, joint_sigmas)

            # for i in range(num_classes):
            #     for j in range(num_classes):
            #         if i != j:
            #             # 最大化非真实类别的 KL 散度
            #             kl_loss -= neg_weight * kl_matrix[i, j]

        except Exception as e:
            print(f"Error calculating KL divergence: {e}")
            continue
    kl_loss = kl_loss / features.size(1)

    return kl_loss / num_samples

def kl_independent_contrastive_loss_v2(features, targets, joint_mus, joint_sigmas, seen_classes, dataset_ordinal, neg_weight=1):
    """
    计算KL散度损失：
    - 正样本：最小化特征与先验类别的KL散度
    - 负样本：最大化真实数据分布之间的KL散度，并根据类别距离调整权重
    """
    kl_loss = 0
    means, vars = accumulate_features(features, targets, [0,1,2])  # 计算各类的均值和方差
    num_samples = features.size(0)
    device = features.device

    # 计算特征与先验分布之间的KL散度矩阵
    kl_matrix_prior = compute_kl_matrix(means, vars, joint_mus, joint_sigmas)
    
    # 计算真实数据分布之间的KL散度矩阵
    kl_matrix_real = compute_kl_matrix(means, vars, means, vars)

    # 正样本损失：最小化特征与对应先验类别的KL散度
    for i in range(3):
        kl_loss += kl_matrix_prior[i, i]
    
    # 负样本损失：最大化真实数据分布之间的KL散度，并根据类别距离调整权重
    for i in range(3):
        for j in range(3):
            if i != j:
                # 计算类别间距离权重：距离越近权重越大
                distance_weight = 1 / abs(i - j)
                # 最大化不同类别真实分布之间的 KL 散度，距离越近的类别施加更大的负权重
                kl_loss -= neg_weight * distance_weight * kl_matrix_real[i, j]

    return kl_loss / num_samples


def train_features(logger, config, model, train_loader, optimizer, criterion, epoch, device):
    """
    只训练KL散度
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0
    targets_list = []
    predicted_list = []
    for idx, (inputs, targets, _) in enumerate(train_loader):
        if not check_class_distribution(targets, config.seen_classes_ordinal):
            continue

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            features, final, targets, _ = model(inputs, targets, mix=False, concat=False, cam=False)
                        
            ordinal_loss = 0
            if torch.cuda.device_count() > 1:
                joint_mus, joint_sigmas, _ = get_joint_priors(model.module.mus, model.module.deltas, model.module.sigmas, config.num_classes)
            else:
                joint_mus, joint_sigmas, _ = get_joint_priors(model.mus, model.deltas, model.sigmas, config.num_classes)

            if config.ordinal_method == 'likelihood':
                ordinal_loss = negative_log_likelihood(features, targets, joint_mus, joint_sigmas, config.num_classes)
            elif config.ordinal_method == 'kl':
                ordinal_loss = kl_independent_loss(features, targets, joint_mus, joint_sigmas, config.seen_classes_ordinal, config.dataset_ordinal)
            elif config.ordinal_method == "multikl":
                ordinal_loss = kl_multi_contrastive_loss(features, targets, joint_mus, joint_sigmas, config.num_classes)

        loss = ordinal_loss
        if loss == 0.0:
            continue

        total_loss += loss.item()
        if not torch.isnan(torch.max(loss)):
            losses.update(loss.item(), len(targets))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_acc, acc = WeightedAccuracy(final, targets, config.num_classes)

        if idx % config.print_freq == 0:
            msg = f'Epoch: [{epoch}][{idx:02}/{len(train_loader):02} Loss {round(losses.val, 4):<10} Acc {acc}]'
            logger.info(msg)

        targets_list.append(targets)
        predicted_list.append(final)
    
    all_targets = torch.cat(targets_list)
    all_predictions = torch.cat(predicted_list)
    acc = Accuracy(all_predictions, all_targets, config.num_classes)

    return total_loss, acc.mean()


def train_features2(logger, config, model, train_loader, optimizer, criterion, epoch, device):
    """
    只训练KL散度
    """
    model.train()
    losses = AverageMeter()
    scaler = torch.cuda.amp.GradScaler()

    total_loss = 0
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            features, final, targets, mixed_features = model(inputs, targets, mix=False, concat=False, cam=False)
                        
            ce_loss = criterion(final, targets)
            kl_loss = torch.Tensor([0.]).to(config.device)

            joint_mus, joint_sigmas, _ = get_joint_priors(model.mus, model.deltas, model.sigmas, config.num_classes)
            means, vars = accumulate_features(features, targets, config.dataset_ordinal)
            for d in range(config.model_features_number):
                try:
                    for i in config.seen_classes_ordinal:
                        post = Normal(means[i][d], vars[i][d].sqrt())
                        prior = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
                        kl_loss += kl_divergence(post, prior).mean()
                except Exception as e:
                    logger.error(e)
                    continue

            kl_loss /= config.model_features_number
            loss = kl_loss + ce_loss

        total_loss += loss.item()
        if not torch.isnan(torch.max(loss)):
            losses.update(loss.item(), len(targets))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_acc, acc = WeightedAccuracy(final, targets, config.num_classes)

        if idx % config.print_freq == 0:
            msg = f'Epoch: [{epoch}][{idx:02}/{len(train_loader):02} Loss {round(losses.val, 4):<10} Acc {acc}]'
            logger.info(msg)
    return total_loss, weighted_acc

def train_classifier2(logger, config, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = AverageMeter()
    num_classes = config.num_classes

    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    for idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            features, final, targets, _ = model(inputs, targets, mix=True, concat=True, cam=False)
            ce_loss = criterion(final, targets)
            kl_loss = 0
            dis_kl_loss_01 = 0
            dis_kl_loss_12 = 0
            joint_mus, joint_sigmas, _ = get_joint_priors(model.mus, model.deltas, model.sigmas, config.num_classes)
            means, vars = accumulate_features(features, targets, config.dataset_ordinal)
            for d in range(config.model_features_number):  # feature 的每个 维度 相互独立, 单独考虑
                try:
                    for i in range(config.num_classes):
                        post = Normal(means[i][d], vars[i][d].sqrt())
                        prior = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
                        kl_loss += kl_divergence(post, prior).mean()

                    post_1 = Normal(means[1][d], vars[1][d].sqrt())
                    post_0 = Normal(means[0][d], vars[0][d].sqrt())
                    post_2 = Normal(means[2][d], vars[2][d].sqrt())

                    prior_1 = Normal(joint_mus[1][d], joint_sigmas[1][1][d].sqrt())
                    prior_0 = Normal(joint_mus[0][d], joint_sigmas[0][0][d].sqrt())
                    prior_2 = Normal(joint_mus[2][d], joint_sigmas[2][2][d].sqrt())
                    dis_kl_loss_01 += kl_divergence(post_1, post_0).mean()
                    dis_kl_loss_01 += kl_divergence(prior_1, prior_0).mean()
                    dis_kl_loss_12 += kl_divergence(post_1, post_2).mean()
                    dis_kl_loss_12 += kl_divergence(prior_1, prior_2).mean()

                except Exception as e:
                    logger.error(e)
                    continue
            loss = ce_loss
            loss = kl_loss + ce_loss - 0.2 * dis_kl_loss_01 - 0.2 * dis_kl_loss_12
            # loss = kl_loss + ce_loss
        total_loss += loss.item()

        if not torch.isnan(torch.max(loss)):
            losses.update(loss.item(), len(targets))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_acc, acc = WeightedAccuracy(final, targets, num_classes)
  
        if idx % config.print_freq == 0:
            msg = f'Epoch: [{epoch}][{idx}/{len(train_loader)}\t Loss {round(losses.val, 4)}\t Acc {acc}]'
            logger.info(msg)
    return total_loss, weighted_acc


def train_classifier(logger, config, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    losses = AverageMeter()

    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    targets_list = []
    predicted_list = []
    
    # 用于存储每个epoch的统计信息
    epoch_stats = {
        'joint_mus': None,  # 联合分布均值
        'joint_sigmas': None,  # 联合分布标准差
        'nablas': None,  # 类间距
        'means': None,  # 类均值
        'vars': None  # 类方差
    }

    for idx, (inputs, targets, _) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast():
            features, final, targets, _ = model(inputs, targets, mix=True, concat=True, cam=False)
            ce_loss = criterion(final, targets)

            ordinal_loss = 0
            if torch.cuda.device_count() > 1:
                mus = model.module.mus
                nablas = model.module.deltas
                sigmas = model.module.sigmas
            else:
                mus = model.mus
                nablas = model.deltas
                sigmas = model.sigmas
                
            joint_mus, joint_sigmas, nablas = get_joint_priors(mus, nablas, sigmas, config.num_classes)
            
            # 记录最后一个batch的统计信息
            epoch_stats['joint_mus'] = joint_mus.detach().cpu()
            epoch_stats['joint_sigmas'] = joint_sigmas.detach().cpu()
            epoch_stats['nablas'] = nablas.detach().cpu()

            means, vars = accumulate_features(features, targets, config.dataset_ordinal) 
            epoch_stats['means'] = means
            epoch_stats['vars'] = vars

            ordinal_loss = kl_independent_contrastive_loss(features, targets, joint_mus, joint_sigmas, config.seen_classes_ordinal, config.dataset_ordinal)
            
            loss = ce_loss + ordinal_loss
            
        total_loss += loss.item()

        if not torch.isnan(torch.max(loss)):
            losses.update(loss.item(), len(targets))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        weighted_acc, acc = WeightedAccuracy(final, targets, config.num_classes)
  
        if idx % config.print_freq == 0:
            msg = f'Epoch: [{epoch}][{idx}/{len(train_loader)}\t Loss {round(losses.val, 4)}\t Acc {acc}]'
            logger.info(msg)
        
        targets_list.append(targets)
        predicted_list.append(final)
    
    # 计算统计量
    stats = {
        'mus_norm': torch.norm(epoch_stats['joint_mus'], p=1, dim=1),  # (3,32) -> (3,)
        'sigmas_norm': torch.norm(torch.stack([
            torch.sqrt(epoch_stats['joint_sigmas'][0,0,:]), 
            torch.sqrt(epoch_stats['joint_sigmas'][1,1,:]), 
            torch.sqrt(epoch_stats['joint_sigmas'][2,2,:])
        ]), p=1, dim=1),  # stack后(3,32) -> norm后(3,)
        'nablas_norm': torch.norm(epoch_stats['nablas'], p=1, dim=1),  # (3,32) -> (3,)
        'means_norm': torch.norm(torch.stack(epoch_stats['means']), p=1, dim=1),  # 先将list转为tensor再计算
        'vars_norm': torch.norm(torch.stack([torch.sqrt(var) for var in epoch_stats['vars']]), p=1, dim=1),  # 先开方转为标准差，再stack成tensor计算一范数
    }

    all_targets = torch.cat(targets_list)
    all_predictions = torch.cat(predicted_list)
    acc = Accuracy(all_predictions, all_targets, config.num_classes)

    return total_loss, acc.mean(), stats


def validate_classifier(logger, config, model, valid_loader, criterion, epoch, device):
    model.eval()

    num_classes = config.num_classes
    total_loss = 0
    corrects, whole = torch.zeros(num_classes), torch.zeros(num_classes)
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            features, final, _, _ = model(inputs, targets, mix=False, cam=False)
            ce_loss = criterion(final, targets)

            total_loss += ce_loss.item()
            batch_size = len(targets)
            predicted_labels = torch.argmax(final, dim=1)
            correct = (predicted_labels == targets).cpu()
            
            # 统计每个类别的正确率
            for j in range(batch_size):
                iclass = targets[j]
                whole[iclass] += 1
                corrects[iclass] += correct[j]
            
            # 收集概率和标签，用于后续计算
            probs = torch.softmax(final, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    class_acc = corrects / whole
    acc = corrects.sum() / whole.sum()
    seen_corrects = [corrects[sc] for sc in config.seen_classes_ordinal]
    seen_whole = [whole[sc] for sc in config.seen_classes_ordinal]
    seen_acc = sum(seen_corrects) / sum(seen_whole)

    unseen_corrects = [corrects[sc] for sc in set(range(config.num_classes)) - set(config.seen_classes_ordinal)]
    unseen_whole = [whole[sc] for sc in set(range(config.num_classes)) - set(config.seen_classes_ordinal)]
    unseen_acc = sum(unseen_corrects) / sum(unseen_whole) if sum(unseen_whole) != 0 else 0

    h = (2 * seen_acc * unseen_acc) / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) != 0 else 0

    weighted_acc = sum(class_acc) / config.num_classes
    
    # 只保留混淆矩阵，移除其他复杂统计信息
    all_probs_tensor = torch.cat(all_probs, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    all_preds = torch.argmax(all_probs_tensor, dim=1)
    
    # 构建混淆矩阵
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for t, p in zip(all_targets_tensor, all_preds):
        confusion_matrix[t.long(), p.long()] += 1
    
    # 计算Micro AUC
    auc_micro = 0
    try:
        # 将张量转换为NumPy数组以计算AUC
        all_labels_np = all_targets_tensor.numpy()  
        all_probs_np = all_probs_tensor.numpy()
        # 使用multi-class one-vs-rest策略计算Micro AUC
        auc_micro = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='micro')
    except Exception as e:
        logger.error(f"计算AUC时出错: {e}")
    
    # 简化统计信息收集
    stats = {
        'class_acc': class_acc.detach().cpu().numpy(),
        'seen_acc': seen_acc,
        'unseen_acc': unseen_acc,
        'harmonic_mean': h,
        'weighted_acc': weighted_acc,
        'confusion_matrix': confusion_matrix.detach().cpu().numpy(),
        'auc_micro': auc_micro
    }

    logger.info('---------------------')
    logger.info('验证集精度为: {0}'.format(acc))
    logger.info(f'验证集加权精度为： {weighted_acc}')
    logger.info('验证集各类精度为: {0}'.format(class_acc))
    logger.info(f'seen acc: {seen_acc}, unseen acc: {unseen_acc}, h: {h}')
    logger.info(f'Micro AUC: {auc_micro}')
    logger.info('---------------------')

    return total_loss, acc, seen_acc, unseen_acc, h, weighted_acc, stats
    

def validate_features(logger, config, model, valid_loader, criterion, epoch, device):
    model.eval()

    num_classes = config.num_classes
    total_loss = 0
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            features, final, _, _ = model(inputs, targets, mix=False, cam=False)
            kl_loss = torch.Tensor([0.]).to(config.device)

            if torch.cuda.device_count() > 1:   
                joint_mus, joint_sigmas, _ = get_joint_priors(model.module.mus, model.module.deltas, model.module.sigmas, config.num_classes)
            else:
                joint_mus, joint_sigmas, _ = get_joint_priors(model.mus, model.deltas, model.sigmas, config.num_classes)

            kl_loss = kl_independent_contrastive_loss(features, targets, joint_mus, joint_sigmas, config.dataset_ordinal, config.dataset_ordinal)
            # kl_loss = kl_multi_loss(features, targets, joint_mus, joint_sigmas)
            # for d in range(config.model_features_number):
            #     try:
            #         for i in range(config.num_classes):
            #             post = Normal(means[i][d], vars[i][d].sqrt())
            #             prior = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
            #             kl_loss += kl_divergence(post, prior).mean()
            #     except Exception as e:
            #         logger.error(e)
            #         continue
            if kl_loss == 0.0:
                continue
            total_loss += kl_loss.item()
    logger.info('---------------------')
    logger.info('验证集KL散度为: {0}'.format(kl_loss))
    logger.info('---------------------')

    return total_loss, 0, 0, 0, 0


def compute_kl_matrix(post_means, post_vars, prior_means, prior_sigmas):
    num_classes = len(post_means)
    kl_matrix = torch.zeros(num_classes, num_classes).cuda()
    for i in range(num_classes):
        for j in range(num_classes):
            for d in range(len(post_means[0])):  # feature 的每个维度相互独立, 单独考虑
                try:
                    if isinstance(post_vars, list):
                        post_dist = Normal(post_means[i][d], post_vars[i][d].sqrt())
                    else:
                        post_dist = Normal(post_means[i][d], post_vars[i][i][d].sqrt())
                    if isinstance(prior_sigmas, list):
                        prior_dist = Normal(prior_means[j][d], prior_sigmas[j][d].sqrt())
                    else:
                        prior_dist = Normal(prior_means[j][d], prior_sigmas[j][j][d].sqrt())
                    kl_div = kl_divergence(post_dist, prior_dist).mean()
                    kl_matrix[i, j] += kl_div
                except Exception as e:
                    print(f"Error calculating KL divergence: {e}")
                    continue
            kl_matrix[i, j] = kl_matrix[i, j] / len(post_means[0])
    return kl_matrix


def debug(logger, config, model, test_loader, pretrain_path):
    model.eval()
    num_classes = config.num_classes
    results = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        corrects, whole = torch.zeros(num_classes), torch.zeros(num_classes)
        for i, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            features, final, targets, mixed_features = model(inputs, targets, mix=True, concat=False)

            probabilities = F.softmax(final, dim=1)  # Predicted probabilities
            _, predicted_labels = torch.max(probabilities, 1)  # Predicted class labels

            all_labels.extend(targets.cpu().numpy())  # Append true labels
            all_probs.extend(probabilities.cpu().numpy())  # Append predicted probabilities
            
            # Collect corrects and whole counts for accuracy
            correct = (predicted_labels == targets).cpu()
            for j in range(len(targets)):
                iclass = targets[j]
                whole[iclass] += 1
                corrects[iclass] += correct[j]
                
                result = {
                    'class_0': probabilities[j][0].item(),
                    'class_1': probabilities[j][1].item(),
                    'class_2': probabilities[j][2].item(),
                    'true_label': targets[j].item(),
                    'predicted_label': predicted_labels[j].item()
                }
                results.append(result)

        acc = corrects / whole  # Calculate per-class accuracy
        seen_corrects = [corrects[sc] for sc in config.seen_classes_ordinal]
        seen_whole = [whole[sc] for sc in config.seen_classes_ordinal]
        seen_acc = sum(seen_corrects) / sum(seen_whole)

        unseen_corrects = [corrects[sc] for sc in set(range(config.num_classes)) - set(config.seen_classes_ordinal)]
        unseen_whole = [whole[sc] for sc in set(range(config.num_classes)) - set(config.seen_classes_ordinal)]
        unseen_acc = sum(unseen_corrects) / sum(unseen_whole) if sum(unseen_whole) != 0 else 0

        h = (2 * seen_acc * unseen_acc) / (seen_acc + unseen_acc) if (seen_acc + unseen_acc) != 0 else 0

        # Compute Micro AUC
        all_labels_flat = np.array(all_labels)  # Flatten the labels (true labels)
        all_probs_flat = np.array(all_probs)  # Flatten the probabilities (predicted scores)

        # Calculate Micro AUC using multi-class one-vs-rest strategy
        auc_micro = roc_auc_score(all_labels_flat, all_probs_flat, multi_class='ovr', average='micro')


        if torch.cuda.device_count() > 1:
            joint_mus, joint_sigmas, _ = get_joint_priors(model.module.mus, model.module.deltas, model.module.sigmas, config.num_classes)
        else:
            joint_mus, joint_sigmas, _ = get_joint_priors(model.mus, model.deltas, model.sigmas, config.num_classes)
        means, vars = accumulate_features(features, targets, config.dataset_ordinal)

        mixed_means = mixed_features.mean(dim=0).type(torch.cuda.FloatTensor)
        mixed_vars = mixed_features.var(dim=0, unbiased=False).type(torch.cuda.FloatTensor)

        prior_prior_kl_matrix = compute_kl_matrix(joint_mus, joint_sigmas, joint_mus, joint_sigmas)
        post_prior_kl_matrix = compute_kl_matrix(means, vars, joint_mus, joint_sigmas)
        post_post_kl_matrix = compute_kl_matrix(means, vars, means, vars)

        mixed_kl_matrix = torch.zeros(2, num_classes).cuda()
        for i in range(num_classes):
            for d in range(len(means[0])):
                post_dist = Normal(means[i][d], vars[i][d].sqrt())
                mixed_dist = Normal(mixed_means[d], mixed_vars[d].sqrt())
                mixed_kl_matrix[0, i] += kl_divergence(mixed_dist, post_dist).mean()
            mixed_kl_matrix[0, i] = mixed_kl_matrix[0, i] / len(means[0])
        for i in range(num_classes):
            for d in range(len(means[0])):
                post_dist = Normal(joint_mus[i][d], joint_sigmas[i][i][d].sqrt())
                mixed_dist = Normal(mixed_means[d], mixed_vars[d].sqrt())
                mixed_kl_matrix[1, i] += kl_divergence(mixed_dist, post_dist).mean()
            mixed_kl_matrix[1, i] = mixed_kl_matrix[1, i] / len(means[0])

        # Log the results
        logger.info('---------------------')
        logger.info('验证集总体精度为: {0}'.format(corrects.sum() / whole.sum()))
        logger.info('验证集各类精度为: {0}'.format(acc))
        logger.info(f'seen acc: {seen_acc}, unseen acc: {unseen_acc}, h: {h}')
        logger.info(f'Micro AUC: {auc_micro}')
        logger.info('先验分布KL散度矩阵:\n{0}'.format(prior_prior_kl_matrix))
        logger.info('后验分布KL散度矩阵:\n{0}'.format(post_post_kl_matrix))
        logger.info('后验分布相对于先验验分布KL散度矩阵:\n{0}'.format(post_prior_kl_matrix))
        logger.info('混合类KL散度矩阵:\n{0}'.format(mixed_kl_matrix))
        logger.info('---------------------')
        
        # Save the results to CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(os.path.dirname(pretrain_path), "prob.csv"), index=False)

    
def cam(logger, config, model, train_loader):

    model.eval()
    num_classes = config.num_classes

    for idx, (inputs, targets, names) in enumerate(train_loader):
        inputs, targets = inputs.to(config.device), targets.to(config.device)

        # with torch.cuda.amp.autocast():
        features, final, targets, _, cams = model(inputs, targets, mix=False, concat=False, cam=True)
        for idx in tqdm(range(len(cams))):
            model.save_visualization(inputs[idx], targets[idx], cams[idx], final[idx], idx, names[idx])
            
            
            
            