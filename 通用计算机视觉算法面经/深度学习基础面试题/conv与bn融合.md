## Conv与BN的融合过程

卷积层的计算公式为$Y=WX + B$,假设batchnorm的均值与方差分别的是$\mu$和 $\sigma^{2}$。线性变换的参数为$\gamma$和$\beta$。求合并后的卷积操作中的$W_{merged}$和$B_{merged}$。





<img src="https://files.mdnice.com/user/6935/84b35586-b6ca-4253-abfc-567e0ef9cf8f.png" style="zoom:50%;" />
