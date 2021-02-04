# -*- coding:utf-8 -*-


def get_parameter_number(net):
    """
    check model structure and compute total amount of parameters
    """
    # print(type(net.parameters()))
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

