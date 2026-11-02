# AMDE 4.0 — Meta-Controller Mathematical Specification

**Status:** RATIFIED  
**Module:** [`amde.controller`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/controller/)  
**Files:** [`base.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/controller/base.py), [`egada.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/controller/egada.py), [`baselines.py`](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/src/amde/controller/baselines.py)

---

## 1. Problem Formulation & Multi-Armed Bandit Model

In AMDE 4.0, autonomous search strategy selection is formalized as a non-stationary Multi-Armed Bandit (MAB) problem. At discrete trial $t \in \{1, \dots, T\}$, the meta-controller chooses a strategy $k_t \in \{1, \dots, K\}$ from the active portfolio $\mathcal{A}$.

The chosen strategy generates a candidate $C_t$, which is evaluated to yield validation score $S(C_t)$, training duration $\Delta t$, and status indicators (novelty, duplicate, failure). The controller receives a scalar reward $R_t \in \mathbb{R}$ and updates its internal posterior beliefs to optimize future strategy allocations.

---

## 2. Mathematical Definition of Controller Feedback & Reward

### 2.1. Historical Defect Analysis ([DEF-02](file:///mnt/c/Users/Admin/Downloads/AMDE_Worktree_v400/docs/architecture/AMDE_4.0_Architecture_Decision_Records.md))
In AMDE 3.0, the incumbent score was initialized to $S^*_0 = -\infty$. Consequently, the very first proposed candidate achieved an improvement:
$$\Delta S_1 = S(C_1) - (-\infty) = +\infty$$
This produced an infinite reward for whichever strategy was pulled first, locking the controller into sampling that arm for $\approx 82\%$ of the total evaluation budget, entirely starving other strategies.

### 2.2. AMDE 4.0 Formally Calibrated Reward Function
AMDE 4.0 resolves this defect through a rigorous, cost-normalized, and bounded reward function:

$$R_t = f(S_t, S^*_{t-1}, \Delta t, \text{flags})$$

1. **Baseline Initialization:** The first candidate $C_1$ establishes the initial baseline: $S^*_1 = S(C_1)$. No improvement is credited: $\Delta S_1 = 0.0$.
2. **Improvement vs Baseline:** For $t > 1$:
   $$\Delta S_t = \max(0.0, S(C_t) - S^*_{t-1})$$
   $$S^*_{t} = \max(S^*_{t-1}, S(C_t))$$
3. **Cost Normalization:** Search efficiency rewards score improvements per unit of compute time (square-root scaling prevents sub-second jitter domination):
   $$\text{Efficiency} = \frac{\Delta S_t}{\sqrt{\Delta t} + 0.1}$$
4. **Absolute Quality Bonus:** A small fraction ($0.05 \cdot S_t$) rewards high absolute scores even when the incumbent is not strictly surpassed.
5. **Penalties for Duplicates and Failures:**
   $$R_t = \begin{cases}
   -0.5 & \text{if candidate is a duplicate} \\
   -1.0 & \text{if candidate evaluation failed / diverged} \\
   \text{clamp}\left(\frac{\Delta S_t}{\sqrt{\Delta t} + 0.1} + 0.05 \cdot S_t, -1.0, 5.0\right) & \text{if candidate is valid and novel}
   \end{cases}$$

---

## 3. Discounted Thompson Sampling with Variance Expansion

To track non-stationary performance across search phases (exploration in early rounds, exploitation in late rounds), AMDE 4.0 implements Discounted Thompson Sampling with Gaussian posteriors.

### 3.1. Posterior Parameter Representation
For each strategy arm $k \in \{1, \dots, K\}$, the controller maintains:
- $\mu_k \in \mathbb{R}$: Estimated mean reward.
- $N_k \in \mathbb{R}^+$: Effective discounted observation count (pseudo-counts).
- Prior values: $\mu_k^{(0)} = 0.20$, $N_k^{(0)} = 1.0$.

### 3.2. Sampling & Strategy Selection
At step $t$, the controller draws a posterior sample $\theta_k$ for each arm from a Gaussian distribution:

$$\theta_k \sim \mathcal{N}\left(\mu_k, \sigma_k^2\right), \quad \text{where } \sigma_k = \sqrt{\frac{1.0}{N_k + 1.0}}$$

The strategy arm with the maximal sample is selected:
$$k^*_t = \arg\max_{k \in \{1, \dots, K\}} \theta_k$$

### 3.3. Posterior Updating & Unchosen Arm Variance Expansion
When arm $k^*$ is pulled and receives reward $R_t$:
1. **Mean Update for Pulled Arm:**
   $$\mu_{k^*} \leftarrow \mu_{k^*} + \frac{R_t - \mu_{k^*}}{N_{k^*} + 1.0}$$
   $$N_{k^*} \leftarrow N_{k^*} \cdot \gamma + 1.0$$
   where discount factor $\gamma = 0.95$.
2. **Variance Expansion on Unchosen Arms:**
   For all unpulled arms $j \neq k^*$:
   $$\mu_j \leftarrow \mu_j \quad \text{(mean preserved)}$$
   $$N_j \leftarrow N_j \cdot \gamma \quad \text{(observation count decays)}$$

As $N_j$ decays by factor $\gamma$ on each step arm $j$ is not chosen, its posterior variance:
$$\sigma_j^2 = \frac{1.0}{N_j \cdot \gamma^\tau + 1.0}$$
monotonically increases. Consequently, unchosen arms become progressively more uncertain, guaranteeing that no strategy arm is ever permanently starved or excluded from exploration.
