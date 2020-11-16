import torch
from torch.nn import Module
from torch.nn.modules.loss import NLLLoss


class WeightedBinaryCrossEntropyLoss(NLLLoss):
    r"""The WeightedBinaryCrossEntropyLoss loss. It is useful to train a binary output maps.

    If provided, the optional argument :attr:`weight` will balance the 1's with respect to the 0's:
    The weight is recommended to be the ratio of 0's in an image.
    Often 90% of the target binary maps is 0 while only 10% is 1.
    Having beta = 0.9 scales the losses on the target-1 pixels with 0.9 and the losses on target-0 pixels with 0.1.

    The `input` given through a forward call is expected to contain
    probabilities for each pixel. `input` has to be a Tensor of size :math:`(minibatch, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    The `target` that this loss expects should be a class index in the range :math:`[1]`
    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - \beta * \sum_{(j \in Y^+)} log(Pr(x_j = 1)) - (1-\beta) \sum_{(j \in Y^-)} log(Pr(x_j = 0))

    where :math:`x` is the probability input, :math:`y` is the target, :math:`\beta` is the balancing weight, and
    :math:`N` is the batch size.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        beta (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target: :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.
    """

    def __init__(self, beta=0.5, reduction='mean'):
        super(NLLLoss, self).__init__()
        self._beta = beta
        assert 0 <= self._beta <= 1
        self._reduction = reduction

    def forward(self, inputs, targets):
        assert targets.max() <= 1
        assert targets.min() >= 0
        assert inputs.max() <= 1
        assert inputs.min() >= 0
        dimension = len(inputs.shape)

        # if an input value == 0, the log value is -inf, where a -1 * -inf == nan.
        epsilon = 1e-3
        unreduced_loss = - self._beta * targets * (inputs + epsilon).log() \
                         - (1 - self._beta) * (1 - targets) * (1 - inputs + epsilon).log()
        # average over all dimensions except the batch dimension
        unreduced_loss = unreduced_loss.mean(dim=tuple([i + 1 for i in range(dimension - 1)]))
        if self._reduction == 'none':
            return unreduced_loss
        elif self._reduction == 'mean':
            return unreduced_loss.mean()
        elif self._reduction == 'sum':
            return unreduced_loss.sum()
        else:
            raise NotImplementedError


class Coral(Module):
    """
    Second order statistics distance minimization from
    paper: https://arxiv.org/pdf/1511.05547.pdf
    """
    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        """
        Calculate the distance between the second order statistics of two sets of samples from two distributions
        :param source: NxD: batch of size N with samples of dimension D
        :param target: NxD: batch of size N with samples of dimension D
        :return: two norm of difference in covariances
        """

        if len(source.shape) == 4:
            size = source.size()
            source = source.view(size[0], size[1] * size[2] * size[3])
            target = target.view(size[0], size[1] * size[2] * size[3])

        # source variance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = (xm.t() @ xm) / source.shape[0]

        # target variance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = (xmt.t() @ xmt) / source.shape[0]

        # frobenius norm between source and target
        return torch.norm(xct - xc, p='fro')


class MMDLossSimple(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if len(x.shape) == 4:
            x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
            y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))
        return ((x.mean(dim=0) - y.mean(dim=0))**2).sum().sqrt()


class MMDLossJordan(Module):
    """
    Seems not to change for different distributions... probably not working.
    MMD2 taking higher order discrepancy between distribution into account
    ref: https://discuss.pytorch.org/t/build-your-own-loss-function-in-pytorch/235/7?u=jordan_campbell
    """
    def __init__(self):
        super().__init__()
        self.alpha = 1/(2 * 3.14)

    def forward(self, x, y):
        B = x.size(0)

        if len(x.shape) == 4:
            x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
            y = y.view(y.size(0), y.size(1) * y.size(2) * y.size(3))

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        K = torch.exp(-self.alpha * (rx.t() + rx - 2 * xx))
        L = torch.exp(-self.alpha * (ry.t() + ry - 2 * yy))
        P = torch.exp(-self.alpha * (rx.t() + ry - 2 * zz))

        beta = (1. / (B * (B - 1)))
        gamma = (2. / (B * B))

        return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)


class MMDLossZhao(Module):
    """
    More information: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    Maximization Mean Discrepancy implementation adjusted from
    https://github.com/ZhaoZhibin/UDTL/blob/master/loss/DAN.py
    paper: https://dl.acm.org/doi/10.5555/3305890.3305909
    """

    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if self.fix_sigma is not None:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):

        if len(source.shape) == 4:
            source = source.view(source.size(0), source.size(1) * source.size(2) * source.size(3))
            target = target.view(target.size(0), target.size(1) * target.size(2) * target.size(3))

        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
