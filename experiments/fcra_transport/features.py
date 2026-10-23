"""Scoped layer1 feature capture/transport without changing model definitions."""

from contextlib import contextmanager

import torch


def layer(model):
    encoder = model[0]
    if not hasattr(encoder, "network") or not hasattr(encoder.network, "layer1"):
        raise ValueError("Transport requires the PACS domainbed ResNet layer1")
    return encoder.network.layer1


def statistics(features):
    mean = features.mean((2, 3), keepdim=True)
    std = features.var((2, 3), unbiased=False, keepdim=True).add(1e-6).sqrt()
    return mean, std.log()


@torch.no_grad()
def describe(model, images):
    captured = []
    handle = layer(model).register_forward_hook(
        lambda module, inputs, output: captured.append(statistics(output)))
    try:
        model.eval()
        raw = model[0](images)
        probabilities = model[1](raw).softmax(1)
    finally:
        handle.remove()
    mean, logstd = captured[0]
    return probabilities, mean.detach(), logstd.detach()


@contextmanager
def transport(model, beta, mean, logstd):
    if beta == 0:
        yield
        return

    def hook(module, inputs, features):
        source_mean, source_logstd = statistics(features)
        target_mean = mean.detach().to(features)
        target_logstd = logstd.detach().to(features)
        new_mean = (1 - beta) * source_mean + beta * target_mean
        new_logstd = (1 - beta) * source_logstd + beta * target_logstd
        return ((features - source_mean) / source_logstd.exp()
                * new_logstd.exp() + new_mean)

    handle = layer(model).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
