import methods.MIML.dsmil.og_dsmil as mil
import torch
import torch.nn as nn
import numpy as np
import os, json, datetime, glob, copy, sys
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, random_split
from methods.MIML.dsmil.focal_loss import FocalLoss
from utils import DiffInfiniteDataset, get_device


def print_save_message(config, save_name, thresholds_optimal):
    if config.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

def save_model(config, fold, run, save_path, model, thresholds_optimal):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(config, save_name, thresholds_optimal)
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump([float(x) for x in thresholds_optimal], f)

    
def print_epoch_info(epoch, config, train_loss_bag, test_loss_bag, avg_score, aucs):
    if config.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
            (epoch, config.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
            (epoch, config.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 

    
def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score
    
def apply_sparse_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
            
def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []

    if len(predictions.shape) == 1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)

    for c in range(num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        
        label = label.flatten()
        prediction = prediction.flatten()


        fpr, tpr, threshold = roc_curve(label, prediction)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)

        try:
            c_auc = roc_auc_score(label, prediction)
            if np.isnan(c_auc):
                print(f"ROC AUC score is not defined for class {c} when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                print(f"ROC AUC score for class {c}: {c_auc}")
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print(f"ROC AUC score is not defined for class {c} when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

    return aucs, thresholds, thresholds_optimal         

def get_malignat_tissue(target):
    return np.any(target, axis=0)

def train(trainloader, milnet: mil.MILNet, criterion, optimizer):
    milnet.train()
    device = get_device()
    total_loss = 0
    for i, (bag, bag_label) in enumerate(trainloader):
        optimizer.zero_grad()
        feats_list = []
        classes_list = []
        bag_predictions = []

        bag = bag.float().to(device)
        bag_label = bag_label.float().to(device)

        for patch in bag:
            feats, classes = milnet.i_classifier(patch)
            feats_list.append(feats)
            classes_list.append(classes)

        bag_feats = torch.cat(feats_list)
        ins_classes = torch.cat(classes_list)

        bag_feats = bag_feats.view(bag.shape[0], bag.shape[1],-1)
        ins_classes = ins_classes.view(bag.shape[0], bag.shape[1], -1)

        for bag_feat, ins_class in zip(bag_feats, ins_classes):
            bag_prediction, A, _ = milnet.b_classifier(bag_feat, ins_class)
            bag_predictions.append(bag_prediction)

        bag_predictions = torch.stack(bag_predictions).view(bag.shape[0], -1)

        max_prediction, _ = torch.max(ins_classes, 1)
        bag_loss = criterion(bag_predictions.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        print('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(trainloader), loss.item()))
    return total_loss / len(trainloader)

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def test(config, testloader, milnet, criterion, thresholds=None, return_predictions=False, validation=False):
    milnet.eval()
    device = get_device()
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for i, (bag, bag_label) in enumerate(testloader):

            feats_list = []
            classes_list = []
            bag_predictions = []

            bag = bag.float().to(device)
            bag_label = bag_label.float().to(device)

            for patch in bag:
                feats, classes = milnet.i_classifier(patch)
                feats_list.append(feats)
                classes_list.append(classes)

            bag_feats = torch.cat(feats_list)
            ins_classes = torch.cat(classes_list)

            bag_feats = bag_feats.view(bag.shape[0], bag.shape[1], -1)
            ins_classes = ins_classes.view(bag.shape[0], bag.shape[1], -1)

            for bag_feat, ins_class in zip(bag_feats, ins_classes):
                bag_prediction, A, _ = milnet.b_classifier(bag_feat, ins_class)
                bag_predictions.append(bag_prediction)

            bag_predictions = torch.stack(bag_predictions).view(bag.shape[0], -1)
            max_prediction, _ = torch.max(ins_classes, 1)


            bag_loss = criterion(bag_predictions.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            
            if validation:
                mode = 'Validation'
            else: mode = 'Test'
            
            print('\r %s bag [%d/%d] bag loss: %.4f' % (mode,i, len(testloader), loss.item()))
            test_labels.append(bag_label.cpu().numpy().astype(int).reshape(-1, config.num_classes))
            if config.average:
                test_predictions.append((torch.sigmoid(max_prediction)+torch.sigmoid(bag_predictions)).cpu().numpy().reshape(-1, config.num_classes))
            else: test_predictions.append(torch.sigmoid(bag_predictions).cpu().numpy().reshape(-1, config.num_classes))
    test_labels = np.concatenate(test_labels, axis=0)
    test_predictions = np.concatenate(test_predictions, axis=0)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, config.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if config.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:
        
        test_labels = test_labels.reshape(-1, config.num_classes)
        test_predictions = test_predictions.reshape(-1, config.num_classes)
        
        
        for i in range(config.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
            
    bag_score = 0
    score_without_first_label = 0
    score_without_last_label = 0
    score_without_first_last_labels = 0
    tissue_malignant = []
    tissue_predicted = []

    num_samples = len(test_labels)

    for i in range(0, num_samples):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score
        score_without_first_label += np.array_equal(test_labels[i, 1:], test_predictions[i, 1:]) 
        score_without_last_label += np.array_equal(test_labels[i, :-1], test_predictions[i, :-1])
        score_without_first_last_labels += np.array_equal(test_labels[i, 1:-1], test_predictions[i, 1:-1])
        
        tissue_malignant.append(get_malignat_tissue(test_labels[i, 1:-1]))
        tissue_predicted.append(get_malignat_tissue(test_predictions[i, 1:-1]))
    
    

    avg_tissue_score = accuracy_score(tissue_malignant, tissue_predicted)
    
    avg_score = bag_score / num_samples
    score_without_first_label = score_without_first_label / num_samples
    score_without_last_label = score_without_last_label / num_samples
    score_without_first_last_labels = score_without_first_last_labels / num_samples
    
    
    print(f"Average score without first class: {score_without_first_label}, without last class: {score_without_last_label}, without first and last class: {score_without_first_last_labels}, average tissue score: {avg_tissue_score}")
    
    if return_predictions:
        return total_loss / len(testloader), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(testloader), avg_score, auc_value, thresholds_optimal                

class OG_DSMIL():
    
    def __init__(self, config):
        self.config = config  
                
    def run(self, trainset:DiffInfiniteDataset, testset:DiffInfiniteDataset, feature_extractor):
        

        trainset.normalize()
        testset.normalize()
        
        trainloader = DataLoader(trainset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
        
        # Split testset into test and validation
        testset, validationset = random_split(testset, [int(0.7*len(testset)), len(testset)-int(0.7*len(testset))])
        
        validationloader = DataLoader(validationset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
        testloader = DataLoader(testset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
        
        
        device = get_device()

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        fold_results = []
        for iteration in range(5):
            print(f"Starting iteration {iteration + 1}.")

            i_classifier = mil.IClassifier(feature_extractor, feature_size=feature_extractor.embed_dim, output_class=self.config.num_classes).to(device)
            b_classifier = mil.BClassifier(input_size=feature_extractor.embed_dim, output_class=self.config.num_classes, dropout_v=self.config.dropout_node, nonlinear=self.config.non_linearity).to(device)
            milnet = mil.MILNet(i_classifier, b_classifier).to(device)
            milnet.apply(lambda m: apply_sparse_init(m))
            criterion = nn.BCEWithLogitsLoss() if self.config.loss == 'BCE' else FocalLoss()
            optimizer = torch.optim.Adam(milnet.parameters(), lr=self.config.lr, betas=(0.5, 0.9), weight_decay=self.config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.num_epochs, 0.000005)

            fold_best_score = -np.inf
            best_ac = 0
            best_auc = [0] * self.config.num_classes
            counter = 0
            best_model = copy.deepcopy(milnet)
            best_thresholds = None
            for epoch in range(1, self.config.num_epochs + 1):
                counter += 1

                train_loss_bag = train(trainloader, milnet, criterion, optimizer) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(self.config, validationloader, milnet, criterion, validation=True)
                print_epoch_info(epoch, self.config, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()
                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(self.config, iteration, run, save_path, milnet, thresholds_optimal)
                    best_model = copy.deepcopy(milnet)
                    best_thresholds = thresholds_optimal
                if counter > self.config.stop_epochs: break
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(self.config, testloader, best_model, criterion, thresholds=best_thresholds)
            fold_results.append((best_ac, best_auc))
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
        # Print mean and std deviation for each class
        print(f"Final results: Mean Accuracy: {mean_ac}")
        for i, mean_score in enumerate(mean_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")