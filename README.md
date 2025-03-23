# jungle_gym_cpp
For practicing libtorch and RL/ML in C++

## Environments

### SnakeEnv

Emulates the snake game with a NN-friendly observation space.

#### Observation space:
Fully observable with each position containing $c$ channels, i.e. a tensor
of shape `[x,y,c]` where `x` and `y` correspond to width/height and $c$ is a channel encoding as follows:
- $c_0$ snake body
- $c_1$ snake head
- $c_2$ food
- $c_3$ wall

#### Rewards:

- REWARD_COLLISION = -1
- REWARD_APPLE = 5
- REWARD_MOVE = -0.05

#### Action space:

- LEFT = 0
- STRAIGHT = 1
- RIGHT = 2

## Policies

### 1. Vanilla Policy Gradient

#### Action sampling
$$
a_t =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a \pi_\theta(a | s_t), & \text{with probability } 1 - \epsilon
\end{cases}
$$

#### Reward
$$ R_t = r_t + \gamma R_{t+1}$$

#### Loss

$$
L_\tau = -\sum_{t=0}^{T-1} \log p(a_t^*) \cdot R_t
$$

where $a_t^*$ is the action corresponding to the maximum probability in $\pi_\theta(a | s_t)$, or the action sampled
randomly in the case of epsilon greedy policies. [^1]

#### Epsilon annealing

$$
\epsilon_t = 0.99^{\frac{cn}{N}}
$$

- $n$ is the current episode index.
- $N$ is the total number episodes. 

The decay terminates when $\epsilon \approx 0.01$ by computing $c = \log_{0.99}(0.01) = 458.211$, for example

### 2. Policy Gradient with entropy regularization

Because the training converges early, entropy regularization might be able to help prevent a feedback loop between 
action sampling (generating the training data) and bias in the policy.

$$
L_{\text{total}} = - \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t | s_t) \cdot R_t + \lambda H(\pi_\theta(a_t | s_t)) \right)
$$

where $R_t$ is computed according to Temporal Difference recurrence relation:

$$
R_t = r_t + \gamma R_{t+1}
$$

and $H(\mathcal{X})$ is the entropy of the action distribution at time $t$:

$$
H(\mathcal{X}) = - \sum_{x \in \mathcal{X}} p(x) \log p(x)
$$

Entropy is maximized when the action distribution emitted by the policy $\pi_\theta(a_t|s_t)$ is uniform, and minimized 
when any value tends toward 1. Entropy regularization therefore rewards exploration, in perhaps a more nuanced way than 
epsilon greedy sampling.

### 3. Actor-critic Policy Gradient with entropy regularization (A2C or AAC)

$$
L_{\text{actor}} = - \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t | s_t) \cdot [R_t - V(s_t)] + \lambda H(\pi_\theta(a_t | s_t)) \right)
$$

$$
L_{\text{critic}} = \frac{1}{2} \sum_{t=0}^{T-1} \left( V(s_t) - \left( R_t + \gamma V(s_{t+1}) \right) \right)^2
$$

### 4. A3C

This implementation of A3C makes use of a specialized, thread safe, parameter optimizer, RMSPropAsync, which 
combines gradients from worker threads to update a shared parameter set. The shared parameter set is then distributed 
back to the workers. It is not lock-free as the original A3C paper claims to be, but it offers a reasonably low 
contention alternative for which each module in the neural network has a separate mutex associated with it. The A3CAgent
class initializes a thread pool of A2CAgents which have a synchronization lambda function, for simplicity and modularity.

[^3]

<p align="center">
  <img src="data/a3c_diagram.drawio.svg" alt="Description">
</p>

## Models

### **ShallowNet**

| **Layer**       | **Dimensions**   |
|-----------------|------------------|
| **input**       | `w*h*4`          |
| **fc**          | 256              |
| **layernorm**   | -                |
| **GELU**        | -                |
| **fc**          | 256              |
| **layernorm**   | -                |
| **GELU**        | -                |
| **fc**          | 256              |
| **layernorm**   | -                |
| **GELU**        | -                |
| **fc**          | `output_size`    |
| **log_softmax** | `output_size`    |


### **SimpleConv**

Densenet with 2 convolution layers and CBAM spatial/channel attention [^4] [^2] 

| **Layer**            | **Dimensions**                                      |
|----------------------|----------------------------------------------------|
| **Input**           | `input_width * input_height * input_channels`      |
| **Conv2D (conv1)**  | `8 filters, kernel=3x3, stride=1, padding=1`       |
| **GELU**            | -                                                  |
| **Concat**          | `input + conv1 output`                              |
| **Conv2D (conv2)**  | `16 filters, kernel=3x3, stride=1, padding=1`      |
| **GELU**            | -                                                  |
| **Concat**          | `input + conv1 output + conv2 output`              |
| **Channel Attention** | `input * channel_attention(input)`               |
| **Spatial Attention** | `channel_attention output * spatial_attention(output)` |
| **Residual Add**    | `input + spatial_attention output`                  |
| **Flatten**         | `input_width * input_height * (input_channels + 8 + 16)` |
| **Fully Connected (fc1)** | `256`                                      |
| **LayerNorm (ln1)** | `256`                                              |
| **GELU**            | -                                                  |
| **Fully Connected (fc2)** | `128`                                      |
| **LayerNorm (ln2)** | `128`                                              |
| **GELU**            | -                                                  |
| **Fully Connected (fc3)** | `output_size`                          |
| **Log Softmax** *(if multiple outputs)* | `output_size`            |


## Results / Demos

### Early implementation of Policy Gradient

An example of a mildly successful Policy Gradient agent trained with entropy regularization. You can see that it has 
converged on a circling behavior for self-avoidance, and it randomly biases its circular motion toward the apple. This
agent was trained with the deprecated 4-directional absolute action space as opposed to the 3-directional relative one.

![Alt Text](data/pg_demo.gif)

### A3CAgent

An example of a slightly more successful A3C agent trained with entropy regularization. It more directly targets the
apples, sometimes to its own detriment. It has a strong left turn bias, which could potentially be fixed with some 
augmentation techniques like mirroring the observation and action space. Trained with:

```
./train_snake --type a3c --gamma 0.9 --learn_rate 0.0001 --lambda 0.07 --n_episodes 60000 --n_threads 24
```

Default episode length is 16 steps. Environments of non-truncated/terminated episodes are carried over into next episode.

![Alt Text](data/a3c_demo.gif)

## References

[^1]: Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. In *Advances in Neural Information Processing Systems*, vol. 12. MIT Press.

[^2]: Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2018). *Densely Connected Convolutional Networks*. Preprint at [https://doi.org/10.48550/arXiv.1608.06993](https://doi.org/10.48550/arXiv.1608.06993).

[^3]: Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. Preprint at [https://doi.org/10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783).

[^4]: Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module*. Preprint at [https://doi.org/10.48550/arXiv.1807.06521](https://doi.org/10.48550/arXiv.1807.06521).

## To do
- Print critic's value estimation for every state during test demo
- plot attention map
- ~~implement a3c (now currently a2c)~~
- ~~Critic network and baseline subtraction~~
- Visualization:
  - basic training loss plot (split into reward and entropy terms)
  - ~~trained model behavior~~
    - save as GIF/video (automatically)
  - action distributions per state
- More appropriate models for encoding observation space
  - ~~CNN (priority)~~
  - RNN
  - GNN <3
- DQN
  - likely important for SnakeEnv, which is essentially [Cliff World](https://distill.pub/2019/paths-perspective-on-value-learning/)
- ~~Abstract away specific NN classes~~
- Exhaustive comparison of methods
- Break out epsilon annealing into simple class (now deprioritized by entropy loss)
