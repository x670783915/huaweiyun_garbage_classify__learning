import torch
import torch.nn as nn

class NMTCriterion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTCritierion, self).__init__()

        self.label_smoothing = label_smoothing
        self.log_soft_max = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        ont_hot = torch.randn(1, num_tokens)
        ont_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot
    
    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = self.log_soft_max(dec_outs)
        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            ont_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                ont_hot = ont_hot.cuda()
            tmp_ = ont_hot.repead(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
        loss = self.criterion(scores, gtruth)
        return loss