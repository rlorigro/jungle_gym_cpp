# jungle_gym_cpp
For practicing libtorch and RL/ML in C++



## SnakeEnv

### Environment description

The observation space consists of the entire playable space, with each position containing three channels, i.e. a tensor
of shape `[x,y,c]` where `x` and `y` correspond to width/height and `c` is an encoding as follows:
- `c_0` snake body
- `c_1` snake head
- `c_2` food

## Notes

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
randomly in the case of epsilon greedy policies.

#### Epsilon annealing

$$
\epsilon_t = 0.99^{\frac{cn}{N}}
$$

- $n$ is the current episode index.
- $N$ is the total number episodes. 

The decay terminates when $\epsilon \approx 0.01$ by computing $c = \log_{0.99}(0.01) = 458.211$, for example

#### Performance

The model occasionally learns first order behaviors that are conducive to survival, limited to very simple strategies 
like always moving up, or walking diagonally {up,right,up,right} to avoid dying for a short time. One session converged on an average 
number of steps in the hundreds, which is large for a 10x10 grid, and clearly requires some avoidance strategies. 
However, training is unstable and often converges early on an extremely low entropy policy.

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

### 3. Actor-critic Policy Gradient with entropy regularization

WIP

$$
L_{\text{actor}} = - \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) \cdot \left[ R_t - V(s_t) \right] - \lambda \sum_{t=0}^{T-1} \sum_{a} \pi_\theta(a | s_t) \log \pi_\theta(a | s_t)
$$

$$
L_{\text{critic}} = \frac{1}{2} \sum_{t=0}^{T-1} \left( V(s_t) - \left( R_t + \gamma V(s_{t+1}) \right) \right)^2
$$

## To do
Implement:
- Break out epsilon annealing into simple class
- Critic network and baseline subtraction
- Visualization:
  - basic training loss plot (split into reward and entropy terms)
  - of trained model behavior (GIF/video)
  - of action distributions per state
- More appropriate model for encoding observation space
  - CNN (priority)
  - RNN
  - GNN
- DQN 
  - likely important for SnakeEnv, which is essentially [Cliff World](https://distill.pub/2019/paths-perspective-on-value-learning/))
- Asynchronous learners

