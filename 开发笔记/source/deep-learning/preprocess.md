# 预处理

## 图像预处理

### 通道位置的变换

由opencv读取的图片其shape默认是channel在后，即(H, W, C)；而pytorch网络一般需要的shape为(C, H, W)，因此可以使用：

```python
<i>.permute(2, 0, 1)
```
