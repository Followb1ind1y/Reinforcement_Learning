# **Function Approximation**

- **Why do we need function approximation?**

    Tabular methods store this knowledge in a table data structure with a size of order $|S| × |A|$ ($|S|$ and $|A|$ denote the sizes of the state space and action spaces, respectively). This is clearly only practical for **moderately-sized** state and action spaces. There are generally three means of representing functions:
    - **Tables of values** (Do not scale)
    - **Analytic expressions** (if we truly had such analytic expressions, there are probably **better** methods of solving our control problem than RL. e.g., control theory, or classical computer science)
    - **Function approximations** (parameterized functions tuned via gradient descent method)
- **Adaptive Functions**

    Let $P$ be some phenomenon of interest in the real world, that takes some input and produces an output. Suppose we do not know what $F$ is. Instead we have a data set of input/output pairs experimentally sampled from $P$:
    
    $$
    DS= {(x^{(i)},t^{(i)})}^{N}_{i=1}
    $$
    
    In supervised learning, $x^{(i)}$ is termed the “data”, $t^{(i)}$ is the target. We wish to approximate the unknown $f$ with an approximation $\hat{f}$ so we can make predictions. We will propose that $\hat{f}$ is some parameterized function, $\hat{f}_{\theta}$ , that can map $D$ to $R$. Our problem is now to find the parameter $\theta_{*}$ such that:
    
    $$
    \hat{f}_{\theta^{*}}: x^{(i)}\to y^{(i)}\approx t^{(i)}
    $$
    
    We can measure the **per-prediction error** (i.e., the error of our prediction of the $i$-th target) via the **squared-error function**:
    
    $$
    PE = \frac{1}{2}(y^{(i)}-t^{(i)})^{2}
    $$
    
    With this we can also assess the **aggregate error** of our predictions over the entire dataset by averaging:
    
    $$
    E = \frac{1}{N}\sum_{\forall i}PE_{i} = \frac{1}{N}\sum_{\forall i}\frac{1}{2}(\hat{f}_{\theta}(x^{(i)})-t^{(i)})^{2}
    $$
    
    The aggregate error, $E$, is parameterized only by $\theta$ since all other components in this expression are constants. Thus $E(\theta)$ defines a **hyper-surface over the space of $\theta$ parameters**. This hyper-surface shows the prediction error for each choice of $\theta$ — our problem is now to traverse this performance hyper-surface to **find the minimum error** setting for $\theta$.
    
- **Gradient Descent**

    Gradient descent is based on the simple idea that we can start from an arbitrary $\theta=\theta_{0}$ and update our estimates in the direction that descends the gradient (”slope”) of $E_{\theta}$. Thus algorithm can be used to search the performance surface for a **local minimum**. This is the gradient descent algorithm, which can be **used to tune a parameterized function** to fit a **general dataset**.
    
    <center><img src="../img/A21.png" width=450px /></center>
    
    - $\alpha>0$ controlling the rate of update on $\theta$. (Setting the α **too high** may cause **instability or oscillation**, while setting α **too low** may **impact the time** to converge)
    - $k_{E}$ sets the threshold on the acceptable aggregate error on our predictor
    - $k_{\nabla}$ sets the threshold on our acceptable criterion for “flatness”.
    
    **Problems with Gradient Descent:**
    
    - Does this mean that we can provide any parameterized function and be assured of getting an appropriate parameterization (i.e. $\theta$)?
    **NO**! **We must have sufficient degrees-of-freedom (i.e., dimensions of $\theta$) to fit a dataset.**
    - Is there a means of computing the acceptable dimensions for $\theta$ (i.e. the degrees of freedom, DOFs)
    **NO! This is called a hyper-parameter, and is set, open-loop, by the engineer.**
    - Is there a guaranty that once the DOFs (i.e., the dimensions of $\theta$) is set, we will get the global optimum?
    **No! There is no guaranty of a global optimum existing, and if we hit a minimum, it may be a local one whose location is sensitive to the initial condition, $\theta_{0}$.**
- **Stochastic Gradient Descent**

    For problems with **very large datasets**, repeated computation of $E$ with the full data set is often impractical. To address this, stochastic gradient descent is used where we, for each iteration, **select a random subset of the dataset** to employ in the computation of $E$. The size of this subset can be decreased right down to using singleton elements of the dataset, randomly selected. This will impact the convergence time, but will will be usable.
- **Parameterized Functions**
    - The elementary parameterized function is the **adaptive linear combiner (ALC)** of linear regression:
        
        $$
        \hat{f} :x \to w^{T}x+b
        $$
        
    - General functions are not linear and so will require more expressiveness than the adaptive linear combiner model. Fortunately there is a simple means of transforming linear regression to enable **polynomial (nonlinear) regression**. We can construct a **feature mapping function**, $\phi$, that transforms $x$ to $\phi(x)$.
        
        <center><img src="../img/A22.png" width=450px /></center>
        
        The **advantages** of this technique is that, if we can identify an appropriate $\phi$, the nonlinear regression problem is merely a **standard application of linear regression** on $\phi(x)$.
        
    - If we can not identify an appropriate $\phi$ — and recall, there are **infinite $\phi$ available** — then we require a technique to which we can resort. The **neural network** is this general technique. We can add a **nonlinearity** $\sigma$ to the ALC’s output as follows:
        
        $$
        z_{1} = w_{1}^{T}x+b \\h_{1}=\sigma(z_{1})
        $$
        
        If $\sigma$ is **differentiable**, then we can compute the gradient as per the requirement of gradient descent, and use this as our object for nonlinear function approximation.
