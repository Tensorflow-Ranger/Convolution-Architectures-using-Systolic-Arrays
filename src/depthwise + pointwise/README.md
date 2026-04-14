# Depthwise + Pointwise (Depthwise Separable Convolution)

## Why This Works

Standard convolution does:
> spatial × channel mixing in one expensive operation

Depthwise separable does:
1. Spatial filtering (cheap) - **Depthwise**
2. Channel mixing (cheap) - **Pointwise**

## Parameter Comparison

*Standard:*
`K² × C_in × C_out`

*Depthwise + Pointwise:*
`K² × C_in + C_in × C_out`

When `K=3`:
This is ~8–9× cheaper.
That’s why MobileNet runs on phones.
