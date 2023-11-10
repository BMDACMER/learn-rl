REINFORCE algorithm---基于策略梯度的强化学习算法

**category**
- model-free：不需要事先知道转换概率。转换概率是对动态环境进行阐述，而这个动态的环境在很多应用场景中很难获取

REINFORCE是基于蒙特卡洛采样的策略梯度方法。也就是agent的采样策略是从开始状态一直到最终状态，是一个完整的轨迹。

参考：https://zhuanlan.zhihu.com/p/426323215