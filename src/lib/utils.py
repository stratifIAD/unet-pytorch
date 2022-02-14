import torch

def dice_coeff_batch(batch_bn_mask, batch_true_bn_mask):
    """ dice_coeff_batch : function that returns the mean dice coeff for a batch of pairs 
    mask, ground truth mask """
    
    def single_dice_coeff(input_bn_mask, true_bn_mask):
        """single_dice_coeff : function that returns the dice coeff for one pair 
        of mask and ground truth mask"""

        # The eps value is used for numerical stability
        eps = 0.0001

        # Computing intersection and union masks
        inter_mask = torch.dot(input_bn_mask.view(-1), true_bn_mask.view(-1))
        union_mask = torch.sum(input_bn_mask) + torch.sum(true_bn_mask) + eps

        # Computing the Dice coefficient
        return (2 * inter_mask.float() + eps) / union_mask.float()

    # Init dice score for batch (GPU ready)
    if batch_bn_mask.is_cuda: dice_score = torch.FloatTensor(1).cuda().zero_()
    else: dice_score = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(batch_bn_mask, batch_true_bn_mask)):
        dice_score +=  single_dice_coeff(inputs[0], inputs[1])
    
    # Return the mean Dice coefficient over the given batch
    return dice_score / (pair_idx + 1)

def metrics(p_n, tp, fp, tn, fn):
    """ Returns accuracy, precision, recall, f1 based on the inputs 
    tp : true positives, fp: false positives, tn: true negatives, fn: false negatives
    For details please check : https://en.wikipedia.org/wiki/Precision_and_recall
    """
    # Computing the accuracy
    accuracy  = (tp + tn) / p_n

    # Computing the precision
    precision =  tp / (tp + fp)

    # Computing the recall
    recall    =  tp / (tp + fn)

    # Computing the f1
    f1        =  2 * tp / (2 * tp + fp + fn)
    return precision, recall, accuracy, f1

def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    # Source of the confusion_matrix function: https://gist.github.com/the-bass
    """
    # Computing the confusion vector
    confusion_vector = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    # Computing the total (p+n)
    p_n = tp + fp + tn + fn

    # Computing the precision, recall, accuracy, f1 metrics
    precision, recall, accuracy, f1 = metrics(p_n, tp, fp, tn, fn)

    return tp/p_n, fp/p_n, tn/p_n, fn/p_n, precision, recall, accuracy, f1 
