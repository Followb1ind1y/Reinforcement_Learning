# **Introduction**

- **What is Reinforcement Learning?**

    Map situations to actions → Maximize a numerical reward signal

- **Reinforcement Learning System?**

    Policy, Reward Signal, Value Function, Model of the Environment
    
    The formulation of the AI control problem: an agent, $M$, interacts with an environment, $E$, by dynamically issuing actions, $A_{t}$, based on environment state feedback, $S_{t}$, and environment reward $R_{t}$. The agent **selects actions** according to policy $\pi(A_{t},S_{t})$. The goal of the agent is to **maximize its return** $G_{t}$.
    
- **The trade-off between exploration and exploitation**
    - **Exploit** their instantaneous knowledge of the environment (already experienced)
    - Refine their knowledge of the environment by experience, which includes **Exploration** of the environment (to make better action selections)
- **Tunable or adjustable Data Structure for a Machine Learning system**
    - Tables of values (numbers) that are adjusted
    - Functions with adjustable coefficients (e.g., neural networks)
- **Machine Learning could be done online and offline**
    - Online Machine Learning system learns while the system is in operation (We are concerned)
    - Offline Machine Learning system has two modes of operation: training mode and production mode

![A1.png](/img/A1.png)

> - $S_{t}$ : State at time $t$   - $A_{t}$: Action selected at time $t$   - $R_{t}$: Reward at time $t$
> 

$$S_{0},A_{0},R_{1},S_{1},A_{1},R_{2},S_{2},A_{2},R_{3},...$$

- **The environment information we have** → The probability of next state and reward given current state and action selected
    
    $$p(s',r|s,a)\doteq \mathrm{Pr}({S_{t}=s',R_{t}=r}|S_{t-1}=s,A_{t-1}=a)$$
    
    The probability that the next state is s’ and reward is r when we choose action a at state s.(Since when we choose action, the next state is not stable and will follow a distribution)
    
    We assume the environment is **state determined** (i.e., a Markov Decision Process)
    
- **Policy:** A mapping from states to probabilities of selecting each possible action.
    - $\pi(a|s)$ → Probability that $A_{t}=a$ if $S_{t} =s$
    
    The probability that the agent will pick action a at state s. (**Choose $A_{t}$ so that $G_{t}$ maximized**)
    
- **State-Value Function** for Policy $\pi$: Expected return when starting in $s$ and following $\pi$
    - $v_{\pi}(s) = E_{\pi}[G_{t}|S_{t}=s]=\sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')]$
- **Action-Value Functio**n for Policy $\pi$: Value of taking action $a$ in state $s$ under policy $\pi$
    - $q_{\pi}(s,a) = E_{\pi}[G_{t}|S_{t}=s,A_{t}=a]= \sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')]$
    
    Note: We can get $q_{\pi}(s,a)$ using $v_{\pi}$
    
    ![A2.png](/img/A2.png)
    
- $\gamma$ is a parameter, $0\leq\gamma\leq1$, called the ***discount rate.*** As $\gamma$ close to 0, the agent will try to maximize immediate rewards. As $\gamma$. close to 1, the agent will care more about the future reward.
    
    $$G_{t}\doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+...=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1},$$
    
- **Optimal Policies** and **Optimal Value Functions**
    
    $$v_{*}(s)\doteq\max_{\pi}v_{\pi}(s), \mathrm{\ for \ all \ } s\in\mathcal{S}$$
    
    Pick the best policy pi which could produce the max output of state-value function.
    
    $$q_{*}(s,a)\doteq\max_{\pi}q_{\pi}(s,a), \mathrm{\ for \ all \ } s\in\mathcal{S}\mathrm{\ and\ } a \in \mathcal{A} $$
    
- **What do we need to directly use Bellman optimality equation to find optimal policy?**
    1. the dynamics of the environment are accurately known; 
    2. computational resources are sufficient to complete the calculation; 
    3. the states have the Markov property.
    
    So in practice, agent **rarely** learns an optimal policy.
