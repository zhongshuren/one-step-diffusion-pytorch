# one-step-diffusion-pytorch

One-step diffusion is of critical importance for building a low-latency interactive generative model.

This repository is a minimal pytorch implementation of a one-step diffusion algorithm, based on two papers, [Shortcut Model](https://arxiv.org/abs/2410.12557) and [Mean Flow](https://arxiv.org/abs/2505.13447). Specifically, we use the sampling strategy of mean flow and the loss function of the shortcut model.

Additionally, there's a slight adjustment to the learning objective. The model now inputs a sample from the Gaussian distribution and directly outputs a generated real data sample.

The evaluation looks like this:
```python
noise = torch.randn(256, 2)
condition = torch.randint(0, 4, (256,))
x = model(noise, c=condition)
```

## Evaluation result from the toy dataset
![](D:\code3\one-step-diffusion-pytorch\assets\result.png)

## TODO list

Integrate with DiT

CFG sampling (not sure if it is implemented correctly)

## Code references

https://github.com/yukara-ikemiya/modified-shortcut-models-pytorch
https://github.com/haidog-yaqub/MeanFlow