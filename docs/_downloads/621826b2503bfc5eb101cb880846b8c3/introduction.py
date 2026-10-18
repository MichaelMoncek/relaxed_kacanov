"""
 Relaxed Kačanov iteration for p-Poisson problem with adaptive error control
 ===========================================================================
"""
# %%
# In this demo, we will first describe and then implement the 
# adaptive error control algorithm for p-Poisson problem using 
# relaxed Kačanov method

# %%
# p-Poisson Problem
# -----------------
# The strong from of the p-Poisson problem is defined as follows: 
#
# .. math::
#    :nowrap:  
#
#    \begin{align*}
#         - \nabla \cdot (|\nabla u|^p \nabla u) = f \quad \text{ on }
#         \Omega\\
#         u  = 0 \quad \text{ on } \partial \Omega
#    \end{align*}

# %% 
# The weak from of the p-Poisson problem is defined as follows:
#
# Find :math:`u` in :math:`W^{1,p}_0(\Omega)` such that the 
# following holds for all :math:`v` in :math:`W^{1,p}_0(\Omega)` 
#
# .. math::
#    :nowrap:
#
#    \begin{align*}
#       \int_{\Omega} |\nabla u|^{p-2} \nabla u \cdot 
#       \nabla v \, dx = \left< f, v \right>
#    \end{align*}
#
# for open and bounded :math:`\Omega \subset \mathbb{R}^n` and
# for :math:`1 < p < \infty`
#

# %%
# The solution of such problem is also the unique minimizer 
# of the energy :math:`\mathcal{J} : W^{1,p}_0(\Omega) \rightarrow 
# \mathbb{R}` defined as:

# %%
# .. math:: 
#    \mathcal{J}(v) := \frac{1}{p} \int_{\Omega} | \nabla v|^p \, dx 
#    - \langle f, v \rangle
#     


# %%
# Relaxed Kačanov iteration
# --------------------------
# The weak formulation can be understood as the following 
# weighted Poisson problem:

# %%
# .. math::
#    :nowrap:
#
#    \begin{align*}
#       \int_{\Omega} a^{p-2} \nabla u \cdot 
#       \nabla v \, dx = \left< f, v \right> \quad
#       \forall v \in W^{1,2}_0{\Omega}
#    \end{align*}
#
# for given :math:`f` and :math:`a : \Omega \rightarrow \mathbb{R}`
# such that :math:`a = |\nabla u|`.
#
# We can than look for the iterative solution :math:`u_n`
# in the following fashion:

# %%
# .. math::
#    :nowrap:
#
#    \begin{align*}
#       \int_{\Omega} |\nabla u_n|^{p-2} \nabla u_{n+1} \cdot 
#       \nabla v \, dx = \left< f, v \right> \quad
#       \forall v \in W^{1,2}_0{\Omega}
#    \end{align*}
#
# which only requires solving a linear problem.
# This problem might be ill-posed in case that the weight
# :math:`|\nabla u_n|^{p-2}` aproaches zero or gets too large.
# 
# A possible solution is the introduction of two relaxation 
# parameters :math:`\epsilon_{-}`, :math:`\epsilon_{+} \in (0, \inf)`,
# :math:`\epsilon_{-} \leq \epsilon_{+}`, which will ensure that 
# :math:`a` will be bounded from above and from zero. We will
# also write 
# :math:`\epsilon = \left[ \epsilon_{-}, \epsilon_{+} \right]`.
# We will therefore replace :math:`a` by the following quantity:

# %%
# .. math::
#    \epsilon_{-} \land a \lor \epsilon_{+} := 
#    \max \left\{ \epsilon_{-}, \min \left\{ a,
#    \epsilon_{+} \right\} \right\}
#

# %%
# Note that this is identity if :math:`a(x) \in \epsilon`
# otherwise :math:`a(x)` gets assigned :math:`\epsilon_{-}` or  
# :math:`\epsilon_{+}` depending on which one is closer.
# For the case :math:`\epsilon_{-} \searrow 0` and 
# :math:`\epsilon_{-} \nearrow \infty` we recover
# the original problem.

# %%
# This is leads us to the following algorithm:

# %%
# .. parsed-literal::
#
#    **Algorithm:** The relaxed :math:`p`-Kačanov algorithm
#
#    **Data:** :math:`f \in (W^{1,p}_0(\Omega))^{*}, 
#    u_0 \in W^{1,2}_0(\Omega)`
#    **Initialize:** :math:`\epsilon_0 = [1, 1]
#    \subset (0, \infty), \, n = 0`
#
#    **while** desired accuracy is not achieved **do**
#        │ Define :math:`a_n := \epsilon_{n,-} \lor 
#        |\nabla v_n| \land \epsilon_{n,+}`
#        │
#        │ Calculate :math:`u_{n+1}` by solving:
#        │ 
#        │     :math:`\int_\Omega a_n^{p-2} \nabla u_{n+1} \cdot
#        \nabla v \, dx = \langle f, v \rangle \quad
#        \forall v \in W^{1,2}_0(\Omega)`;
#        │ 
#        │ Choose new relaxation interval :math:`\epsilon_{n+1}
#        \supset \epsilon_{n}`;
#        │ Increase :math:`n` by 1;
#    **end**

# %%
# this algorithm is called the Relaxed Kačanov method.
# A justification of why it converges can be find 
# for example here :cite:`diening2022`.

# %% 
# We will now discuss how to choose :math:`\epsilon_{n}`.
# In :cite:`diening2022` the following choice of :math:`\epsilon_n` 
# is suggested:
#
# :math:`\epsilon_n := \left[(n+1)^{-\alpha}, (n+1)^{\beta}\right]`
# with :math:`\alpha, \beta > 0` and 
# :math:`\alpha + \beta < 1 / (2-p)`
# 
# However it is possible to do even better by utilizing
# adaptive error estimates, which we will discuss in 
# the following section.

# %%
# Error estimates
# ---------------
# We will start by defining the following energy:
 

# %%
# .. math::
#    \mathcal{J}(v, a) = \int_{\Omega} \frac{1}{2} a^{p-2}
#    | \nabla v |^2 + \left( \frac{1}{p} - \frac{1}{2} \right) 
#    a^p \, dx - \langle f, v \rangle.
#

# %%
# Note that this energy does coincide with the original energy 
# for :math:`a = | \nabla v|`.
# This newly defined energy is also convex with respect to 
# :math:`(a, v)` since :math:`(t,a) \mapsto \frac{1}{2} 
# a^{p-2} t^2` is convex on :math:`[0, \infty), 1 < p \leq 2` and 
# the addition of two convex functions is still a convex function.
# Minimization of this energy energy leads to the following problem:

# %%
# .. math::
#    - \text{div}(a^{p-2} \nabla v) = f

# %%
# However this problem loses its ellipticity as
# :math:`a(x) \rightarrow 0` and :math:`a(x) \rightarrow \infty`.
#
# The remedy is to restrict :math:`a` to
# :math:`\epsilon = \left[ \epsilon_{-}, \epsilon_{+} \right]
# \subset (0, \infty)`
# We can than define another relaxed energy:

# %%
# .. math::
#    \mathcal{J}_{\epsilon}(v) := \mathcal{J}(v, \epsilon_{-} \lor
#    | \nabla v | \land \epsilon_{+}) = \underset{a: \epsilon_{-} 
#    \leq a \leq \epsilon_{+}}{\min} \mathcal{J}(v, a).
#

# %%
# The relaxed energy :math:`\mathcal{J}_{\epsilon}(v)` is 
# monotonically decreasing with respect to :math:`\epsilon`
# i.e increasing the relaxation interval :math:`\epsilon` 
# decreases the energy :math:`\mathcal{J}_{\epsilon}(v)`.
# We can also write :math:`\mathcal{J}_{\epsilon}` explicitely
# as:

# %%
# .. math::
#
#    \begin{align*}
#       \mathcal{J}_{\epsilon}(v) = \int_{\Omega} 
#       \kappa_{\epsilon}(| \nabla v|) \, dx - \langle f, v \rangle
#    \end{align*}
#

# %%
# with :math:`\kappa_{\epsilon} : \mathbb{R}_{\geq 0} \rightarrow 
# \mathbb{R}` given by

# %%
# .. math::
#    \begin{align*}
#       \kappa_{\epsilon}(t) := 
#       \begin{cases}
#          \frac{1}{2} \epsilon_{-}^{p-2} \, t^2 + 
#          \left( \frac{1}{p} - \frac{1}{2}\right)\epsilon_{-}^p
#          \quad \text{ for } t \leq \epsilon_{-} \\
#          \frac{1}{p} t^p \hspace{9em}
#          \text{ for } t \in [\epsilon_{-}, \epsilon_{+}]\\
#          \frac{1}{2} \epsilon_{+}^{p-2} \, t^2 + 
#          \left( \frac{1}{p} - \frac{1}{2}\right)\epsilon_{+}^p
#          \quad \text{ for } t \geq \epsilon_{+} \\
#       \end{cases}
#    \end{align*}

# %%
# Next let us define

# %%
# .. math::
#    \phi_{\epsilon}(t) := \kappa(t) - \kappa(0)

# %%
# Explicitely

# %%
# .. math::
#    \begin{align*}
#       \phi_{\epsilon}(t) := 
#       \begin{cases}
#           \frac{1}{2} \epsilon_{-}^{\, p-2} t^2 \hspace{7em}
#           \text{ for } t \leq \epsilon_{-}^{p-1} \\
#           \frac{1}{p'}  t^{\, p'}  - C \epsilon_{-}^p \hspace{5.25em}
#           \text{ for } 
#           \epsilon_{-}^{\, p-1} \leq t \leq \epsilon_{+}^{p-1} \\ 
#           \frac{1}{2} \epsilon_{+}^{\, p-2} t^2 +
#           C(\epsilon_{+}^p - \epsilon^p_{-}) \quad 
#           \text{ for } t \geq \epsilon_{+}^{p-1} \\
#       \end{cases}
#    \end{align*}

# %%
# which is an example of so-called N-function.
# Observe that 


# %%
# .. math::
#    \frac{\phi'_{\epsilon}(t)}{t} = \left(\epsilon_{-} \lor
#    t \land \epsilon_{+} \right)^{p-2}

# %%
# Next we will compute the convex conjugate of :math:`\phi`
# which we will need later.

# %%
# .. math::
#    \begin{align*}
#       \phi_{\epsilon}^{*}(t) := 
#       \begin{cases}
#           \frac{1}{2} \epsilon_{-}^{2-p} t^2 \hspace{7em} 
#           \text{ for } t \leq \epsilon_{-}^{p-1} \\
#           \frac{1}{p'}  t^{p'}  + C \epsilon_{-}^p \hspace{5.25em}
#           \text{ for }
#           \epsilon_{-}^{p-1} \leq t \leq \epsilon_{+}^{p-1} \\ 
#           \frac{1}{2} \epsilon_{+}^{2-p} t^2 -
#           C(\epsilon_{+}^p - \epsilon^p_{-}) \quad 
#           \text{ for } t \geq \epsilon_{+}^{p-1} \\
#       \end{cases}
#    \end{align*}

# %% 
# where :math:`p' \in [1, \infty]` is the usual conjugate exponent of :math:`p` s.t.
# :math:`1/p + 1/p' = 1` and :math:`C = (1/p - 1/2)`

# %%
# Next we define for all :math:`a \geq 0` the *shifted* N-function
# :math:`\varphi_a` by its derivative 


# %%
# .. math::
#    \begin{align*}
#       \varphi'_a(t) := \frac{\varphi'(t \lor a)}{t \lor a} t
#       \quad \text{ and } \quad
#       \varphi_a(t) = \int_0^t \varphi'_a(\tau) \, d\tau
#    \end{align*}

# %% 
# With all this prepared we can explicitely write down
# :math:`(\phi_{\epsilon, a})^{*}` needed for the 
# discretization and linearization error estimators. 

# %%
# **Warning!** 
# We have collision of notation as the lower index means both 
# :math:`\phi_{\epsilon}` --- a funcition defined above
# and the truncation of a function.
# Therefore we will write :math:`\phi_{\epsilon, a}` which
# means :math:`\phi_{\epsilon}` truncated by :math:`a`.

# %% 
# To make our lives easier we can use
# **Lemma 33** of :cite:`diening2022`:

# %%
# .. math::
#    (\varphi_a)^{*} = (\varphi^{*})_{\varphi'(a)}

# %% 
# and the following property of :math:`\varphi`:

# %%
# .. math::
#    (\varphi^{*})'(t) = (\varphi')^{-1}(t)

# %%
# which is a direct consequence of the definition of the 
# convex conjugate.
# Defining :math:`C(a) = (\epsilon_{-} \lor a \land \epsilon_{+})`
# we can write:

# %%
# .. math::
#    \begin{align*}
#       \phi_{\epsilon, a}^{*}(t) := 
#       \begin{cases}
#           \frac{1}{2} C(a)^{2-p} t^2 \hspace{7.5em} 
#           \text{ for } t \leq C(a)^{p-2}a \\
#           \phi^{*}(t) + \phi(a) - \frac{1}{2} C(a)^{p-2}a^2
#           \quad \text{ for } t \geq C(a)^{p-2}a
#       \end{cases}
#    \end{align*}

# %%
# Finally we are ready to introduce the adaptive version 
# of the previous algorithm.
# This method will capture the effect of truncation of
# :math:`a`, the discretization error and the fixpoint
# iteration error.
# Lets start by defining the following error measures:

# %%
# **Upper truncation error**
#
# .. math::
#    \eta_{\epsilon_{+}}^2(v_n) := \mathcal{J}_{\epsilon_n}(v_n) -
#    \mathcal{J}_{(\epsilon_{n,-}, \infty)}(v_n)

# %%
# which measures the effect of the upper truncation bound 
# :math:`\epsilon_{n,+}`.  

# %%
# **Lower truncation error**
#
# .. math::
#    \eta_{\epsilon_{-}}^2(v_n) := \mathcal{J}_{\epsilon_n}(v_n) -
#    \mathcal{J}_{(0, \epsilon_{n,+})}(v_n)

# %%
# which measures the effect of the lower truncation bound 
# :math:`\epsilon_{n,-}`.

# %%
# **Discretization error**
#
# We use the optimal estimator :cite:`diening2008` with 
# the :math:`\phi_{\epsilon_n}` to estimate discretiztion
# error on mesh element :math:`T`:


# %%
# .. math::
#    \eta_h^2(v_n) := \int_T (\phi_{\epsilon_n, | \nabla v_n|})^{*}
#    (h_T |f|) \, dx \: + \: \sum_{F \subset \partial T} h_F
#    \int_F | \llbracket V(v_n) \rrbracket |_F^2 \, ds

# %%
# **Linearization error**
#
# To measure the linearization error we will use the following 
# estimator proposed in :cite:`diening2022`


# %%
# .. math::
#    \eta_{lin}^2(v_n) := \int_{\Omega} 
#    (\phi_{\epsilon, | \nabla v_n|})^{*}
#    \left( \frac{\phi'_{\epsilon}(| \nabla v_{n-1}|)}{
#    \nabla v_{n-1}|} | \nabla (v_{n-1} - v_{n}| \right) \, dx

# %% 
# with help of these estimators we can finally write down 
# the full adaptive algorithm for the relaxed p-Kačanov iteration.

# %%
# .. parsed-literal::
#
#    **Algorithm:** The relaxed :math:`p`-Kačanov algorithm
#
#    **Data:** :math:`f \in (W^{1,p}_0(\Omega))^{*},
#    u_0 \in W^{1,2}_0(\Omega)`
#    **Initialize:** :math:`\epsilon_0 = [1, 1] 
#    \subset (0, \infty)`, :math:`n = 0`
#
#    **while** desired accuracy is not achieved **do**
#        │ Define :math:`a_n := \epsilon_{n,-} \lor
#        |\nabla u_n| \land \epsilon_{n,+}`
#        │
#        │ Calculate :math:`u_{n+1}` by solving
#        │     :math:`\int_\Omega a_n^{p-2} \nabla u_{n+1}
#        \cdot \nabla v \, dx = \langle f, v \rangle`
#        │
#        │ Increase the total cost by the current number of degrees of freedom
#        │
#        │ Compare the estimators :math:`\eta^2_{\epsilon+}(u_n),
#        \eta^2_{\epsilon-}(u_n), \eta^2_h(u_n), \eta^2_{lin}(u_n)`:
#        │     **if** :math:`\eta^2_{\epsilon+}` is largest: 
#        │        :math:`\epsilon_{n+1,+} := 1.25\,\epsilon_{n,+}`
#        │     **if** :math:`\eta^2_{\epsilon-}` is largest: 
#        │        :math:`\epsilon_{n+1,-} := 0.80\,\epsilon_{n,-}`
#        │     **if** :math:`\eta^2_h` is largest: 
#        │        refine the mesh by Dörfler marking
#        │     **if** :math:`\eta^2_{lin}` is largest: 
#        │        set :math:`n := n+1`
#    **end**

