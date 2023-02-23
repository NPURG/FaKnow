from torch.autograd import Function


class _ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 只需要对输入的x返回loss，其他的返回None
        # 详见 https://zhuanlan.zhihu.com/p/263827804
        return grad_output * -ctx.lambd, None
