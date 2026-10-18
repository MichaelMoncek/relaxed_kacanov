"""
p-Poisson solver with adaptive error control in Firedrake
by Michael Moncek
"""

# %%
# Implementation
# ^^^^^^^^^^^^^^

# %%
# Lets begin by importing the necessary libraries:

import firedrake as fd
from firedrake import *
import numpy as np
from netgen.occ import *

# %%
# This is a safeguard as inner(v,v) might return negative
# value because of FPA

def norm_safe(v):
    return sqrt(max_value(inner(v,v), 0.0))

# %%
# We will write :math:`\kappa_{\epsilon}(t)` in slightly convoluted 
# way to avoid problems with FPA

def kappa(t, p, eps_minus=0.0, eps_plus=float("inf")):
    """
    kappa_eps(t), Sec. 2 of (DFW22)
    """
    def j(eps):
        return (0.5*eps**(p-2)*t**2 + (1/p-0.5)*eps**p)

    expr = 1/p * t**p

    if float(eps_minus) > 0.0: # don't truncate for small eps
          expr = conditional(le(t, eps_minus), j(eps_minus), expr)

    if float(eps_plus) < float("inf"): # don't truncate for large eps
          expr = conditional(ge(t,eps_plus), j(eps_plus), expr)

    return expr

# %%
# Energy :math:`\mathcal{J}_{\epsilon}`:

def J_eps(v, f, p, eps_minus=0.0, eps_plus=float("inf")):
    """
    Eq 2.2, Sec. 2 of (DFW22)
    """
    grad_norm = norm_safe(grad(v))
    return kappa(grad_norm, p, eps_minus, eps_plus)*dx - f*v*dx

# %%
# Next let us define the clamp function

# %%
# .. math::
#    \text{clamp}(t, \epsilon_{-}, \epsilon_{+}) :=
#    \epsilon_{-} \land t \lor \epsilon_{+}
# which will be very useful throughout.

def clamp(t, eps_minus, eps_plus):
    """
    max{eps_minus, min{t, eps_plus}}
    """
    return max_value(eps_minus, min_value(t, eps_plus))

# %%
# Define

# %%
# .. math::
#    \phi_{\epsilon}(t) := \kappa(t) - \kappa(0)

def phi_eps(t, eps_minus, eps_plus, p):
    """
    Equation (3.1) in (DFW22), 
    phi_eps(t) = kappa_eps(t) - kappa_eps(0)
    """
    const = (1/p - 0.5)
    low_branch  = 0.5 * eps_minus**(p-2) * t**2
    mid_branch  = (1/p) * t**p - const * eps_minus**p
    high_branch =  (0.5 * eps_plus**(p-2) * t**2 
                    + const * (eps_plus**p - eps_minus**p))

    return conditional(le(t, eps_minus), low_branch, 
                       conditional(le(t, eps_plus), 
                                   mid_branch, high_branch))

# %%
# We will also need the convex conjugate (Legendre transform,
# Fenchel conjugate) of :math:`\phi_{\epsilon}`: 

def phi_eps_conj(t, eps_minus, eps_plus, p):
    """
    Convex conjugate of Eq (3.1) in (DFW22)
    """
    pp = p / (p-1) # conjugate exponent of p s.t. 1/p + 1/pp = 1 
    const = (1/p - 0.5)
    low_branch  = 0.5 * eps_minus**(2-p) * t**2
    mid_branch  = (1/pp) * t**pp + const * eps_minus**p
    high_branch = (0.5 * eps_plus**(2-p) * t**2 
                   - const * (eps_plus**p - eps_minus**p))

    return conditional(le(t, eps_minus**(p-1)), low_branch,
                       conditional(le(t, eps_plus**(p-1)),
                                   mid_branch, high_branch))

# %%
# and we will need :math:`(\phi_{\epsilon, a})^{*} =
# ((\phi_{\epsilon})_{a})^{*}` - 
# the convex conjugate of the shift of :math:`\phi_{\epsilon}`

def phi_eps_shift_conj(t, a, eps_minus, eps_plus, p):
    """
    Convex conjugate of the shifted N-function phi_eps
    Eq (3.1) in (DFW22), used for the error estimators
    in section 6 of (DWF22)
    """
    Ca = clamp(a, eps_minus, eps_plus)
    b = Ca**(p-2) * a
    low_branch  = 0.5 * Ca**(2-p) * t**2
    high_branch = (phi_eps_conj(t, eps_minus, eps_plus, p) + 
                   phi_eps(a, eps_minus, eps_plus, p) -
                   0.5 * Ca**(p-2) * a**2)

    return conditional(le(t, b), low_branch, high_branch)

# %%
# Now, we are ready to implement the following error measures:

# %%
# **Upper truncation error**
#
# .. math::
#    \eta_{\epsilon_{+}}^2(v_n) := \mathcal{J}_{\epsilon_n}(v_n) -
#    \mathcal{J}_{(\epsilon_{n,-}, \infty)}(v_n)

# %%
# which measures the effect of the upper truncation bound 
# :math:`\epsilon_{n,+}`.  

def eta2_plus(v, f, eps_minus, eps_plus, p):
    """
    Upper truncation error squared
    """
    return (J_eps(v, f, p, eps_minus, eps_plus) -
            J_eps(v, f, p, eps_minus, float("inf")))

# %%
# **Lower truncation error**
#
# .. math::
#    \eta_{\epsilon_{-}}^2(v_n) := \mathcal{J}_{\epsilon_n}(v_n) -
#    \mathcal{J}_{(0, \epsilon_{n,+})}(v_n)

# %%
# which measures the effect of the lower truncation bound 
# :math:`\epsilon_{n,-}`.

def eta2_minus(v, f, eps_minus, eps_plus, p):
    """
    Lower truncation error squared
    """
    return (J_eps(v, f, p, eps_minus, eps_plus) -
            J_eps(v, f, p, 0.0, eps_plus)) 

# %%
# **Discretization error**
#
# We use the optimal estimator :cite:`diening2008` with 
# the :math:`\phi_{\epsilon_n}` to estimate discretiztion
# error on mesh element :math:`T`:

# %%
# .. math::
#    \eta_{h,T}^2(v_n) := \int_T (\phi_{\epsilon_n, | \nabla v_n|})^{*}
#    (h_T |f|) \, dx \: + \: \sum_{F \subset \partial T} h_F
#    \int_F | \llbracket V(v_n) \rrbracket |_F^2 \, ds


# %%
# where :math:`\llbracket v \rrbracket = v_+ - v_-`  is 
# the jump of :math:`v`. 
# The quantities :math:`h_F, h_T` are the characteristic
# lengths of a facet and triangle (element) respectively and 
# :math:`V(v)` is defined as:

# %%
# .. math::
#    V_{\epsilon}(v) = 
#       \begin{cases}
#           \left( \frac{\phi'_{\epsilon}(|v|)}{|v|} \right)^{1/2} v
#           \quad \text{ for } P \neq 0\\
#           0 \hspace{5.75em} \text{ for } P = 0
#       \end{cases}


def V_eps(v, eps_minus, eps_plus, p):
    """
    Definition 7 (DFW22)
    """
    t = norm_safe(v)
    s = clamp(t, eps_minus, eps_plus)
    return s**(p/2 -1) * v

# %%
# Here we use a neat trick to transform
# the estimator into the following problem
# by working on a piecewise constant DG0 space, for details see: 
# https://www.firedrakeproject.org/demos/adaptive_multigrid.py.html

def eta2_disc(mesh, uh, f, eps_minus, eps_plus, p):
    """
    Discretization error from (DK08)
    """
    h_T = CellDiameter(mesh) # characteristic length of Triangle
    h_F = FacetArea(mesh) # characteristic length of Facet
    vol = CellVolume(mesh)

    Q = FunctionSpace(mesh, "DG", 0)
    q = TestFunction(Q)
    phi = TrialFunction(Q)

    grad_uh = grad(uh)
    norm_grad_uh = norm_safe(grad_uh)
    eta2 = Function(Q)

    # compute cellwise error estimator
    V_grad = V_eps(grad_uh, eps_minus, eps_plus, p)
    jump_V_grad = jump(V_grad)
    a = inner(phi, q / vol) * dx
    # t = h_T * norm(f)
    t = h_T * abs(f)
    integrand = phi_eps_shift_conj(t, norm_grad_uh,
                                   eps_minus, eps_plus, p)
    L = (integrand * q * dx +
    (inner(jump_V_grad, jump_V_grad) * 2 * avg(q * h_F)) * dS)

    sp = {"mat_type": "matfree", "ksp_type": "preonly", "pc_type": "jacobi"}
    solve(a == L, eta2, solver_parameters=sp)

    # compute eta from eta^2
    eta = Function(Q, name="eta").interpolate(sqrt(eta2))
    with eta.dat.vec_ro as eta_:
        error_disc = eta_.norm()**2
    return eta, error_disc


# %%
# **Linearization error**
#
# To measure the linearization error we can either choose the
# following estimator proposed in :cite:`diening2022`

# %%
# .. math::
#    \eta_{lin}^2(v_n) := \int_{\Omega} 
#    (\phi_{\epsilon, | \nabla v_n|})^{*}
#    \left( \frac{\phi'_{\epsilon}(| \nabla v_{n-1}|)}{
#    \nabla v_{n-1}|} | \nabla (v_{n-1} - v_{n}| \right) \, dx

# %% 
# or we can work with the natural distance which
# is the (integrated) Bregman divergence
# between :math:`\nabla v_n - \nabla v_{n-1}` associated 
# with :math:`\phi_{\epsilon}` :cite:`alexei2026`:

# %%
# .. math::
#    \eta_{lin}^2(v_n) := \int_{\Omega}
#    \left[ \phi_{\epsilon}(|\nabla v_{n-1}|) 
#    - \phi_{\epsilon}(|\nabla v_{n}|) - 
#    C(|\nabla v_n|)^{p-2} \nabla v_n \cdot (\nabla v_{n-1} - 
#    \nabla v_{n})
#    \right] dx

# %%
# where :math:`C(a)` is the clamp function defined above.
# We will choose the first definition to work with,
# however we have implemented both for good measure.
def eta2_lin(u, u_, eps_minus, eps_plus, p):
    """
    Linearization error proposed in (Gaz26)
    """
    norm_grad_u_ = norm_safe(grad(u_)) 
    norm_grad_u = norm_safe(grad(u)) 
    a = clamp(norm_grad_u, eps_minus, eps_plus)**(p-2)
    energy_diff = (phi_eps(norm_grad_u_, eps_minus, eps_plus, p)
                   - phi_eps(norm_grad_u, eps_minus, eps_plus, p)
    )
    correction = a * inner(grad(u), grad(u_) - grad(u))

    return (energy_diff - correction)*dx

# %%
def eta2_lin_kac(u, u_, eps_minus, eps_plus, p):
    """
    Linearization error proposed in (DFW22)
    """
    a = norm_safe(grad(u_))       
    C = clamp(a, eps_minus, eps_plus)**(p-2)  
    s = C * norm_safe(grad(u_) - grad(u))
    return phi_eps_shift_conj(s, a, eps_minus, eps_plus, p) * dx

# %%
# Helper functions
# ----------------

# %%
# This function does the 'Kačanov step' i.e.
# the following part of the algorithm:


# %%
# .. parsed-literal::
#        │ Calculate :math:`u_{n+1}` by solving:
#        │ 
#        │     :math:`\int_\Omega a_n^{p-2} \nabla u_{n+1} \cdot
#        \nabla v \, dx = \langle f, v \rangle \quad
#        \forall v \in W^{1,2}_0(\Omega)`;

def kacanov_step(u_, f, bc, V, atm, eps_minus, eps_plus, p):
    """
    Do one Kacanov step i.e. Solve the linear poisson problem
    -div(a*grad(u)) = f with coefficient matrix
    a = clamp(grad(u_), eps_minus, eps_plus)
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = DirichletBC(V, 0.0, "on_boundary")
    un = Function(V, name="u")

    grad_norm = norm_safe(grad(u_))
    a = clamp(grad_norm, eps_minus, eps_plus)**(p-2)
    F = a*inner(grad(u), grad(v))*dx - f*v*dx 
    
    problem = fd.LinearVariationalProblem(lhs(F), rhs(F), un, bc)
    sp = {"ksp_type": "preonly", "pc_type": "lu",
         "pc_factor_mat_solver_type": "mumps"}
    solver = fd.LinearVariationalSolver(problem,
                                        solver_parameters=sp)

    solver.set_transfer_manager(atm)
    solver.solve()
    return un

# %%
# This function computes the estimated errors

# %%
# .. parsed-literal::
#    │ :math:`\eta^2_{\epsilon+}(u_n),
#    \eta^2_{\epsilon-}(u_n), \eta^2_h(u_n), \eta^2_{lin}(u_n)`:

def estimate_errors(mesh, u, u_, f, eps_minus, eps_plus, p):
    """
    Returns (eta, errors) - discretization error on each element
    and dictionary of the linearization, plus, minus and
    discretization error sizes
    """
    error2_minus = assemble(eta2_minus(u, f, eps_minus, eps_plus, p))
    error2_plus  = assemble(eta2_plus(u, f, eps_minus, eps_plus, p))
    # error2_lin   = assemble(eta2_lin(u, u_, eps_minus, eps_plus, p))
    error2_lin   = assemble(eta2_lin_kac(u, u_,
                                         eps_minus, eps_plus, p)) 
    eta, error2_disc = eta2_disc(mesh, u, f, eps_minus, eps_plus, p)

    errors = {
              "lin" : error2_lin, "plus" : error2_plus,
              "minus" : error2_minus,"disc" : error2_disc
              }
    return eta, errors

# %%
# Update the state according to the largest error estimate
def adapt(errors, eta, amh, eta_refinement, n,
          refinement_num, max_refinements,
          eps_minus, eps_plus, eps_tol=1e6):
    """
    Update the state according to the largest error estimate
    """
    largest_error = max(errors, key=errors.get)

    if largest_error == "lin":
       n += 1
    elif largest_error == "plus":
       if (abs(float(eps_plus)) <= eps_tol):
           eps_plus.assign(1.25*eps_plus)
       else:
           n += 1
    elif largest_error == "minus":
       if (abs(float(eps_minus)) >= 1/eps_tol):
           eps_minus.assign(0.80*eps_minus)
       else:
           n += 1
    else: # "disc"
       if (refinement_num < max_refinements):
           amh.adapt(eta, eta_refinement)
           refinement_num += 1
           return n, refinement_num, False
       else:
           return n, refinement_num, True
    return n, refinement_num, False

# %%
# The main function that handles the logic.
# The code used for graphing, plotting etc.
# was created with Claude.
def p_poisson(p, make_f, make_bc, make_u_exact,
              mesh, amh, atm, tol=1e-8):
    """
    Solves -div(|grad(u)|^(p-2) * grad(u)) = f on mesh
    with boundary conditions given by 'bc'.
    """
    p = Constant(p)
    eps_plus  = Constant(1.0)
    eps_minus = Constant(1.0)
    eps_tol   = 1e6
    eta_refinement = 0.5
    refinement_num = 0
    max_refinements = 15
    n, n_max  = 0, 1000
    iter_num = 0
    cost = 0
    history = []

    out_file = fd.VTKFile("results/u_level_0.pvd")
    u_ = None
    while (n < n_max):
        mesh = amh[-1]
        # set up exact solution, f and boundary conditions
        V = FunctionSpace(mesh, "Lagrange", 1)
        u_exact = make_u_exact(mesh)
        f = make_f(mesh)
        bc = make_bc(V)

        if u_ is None:
           u_ = Function(V)
        elif u_.function_space() != V:
           u_new = Function(V)
           atm.prolong(u_, u_new)
           u_ = u_new

        u = kacanov_step(u_, f, bc, V, atm, eps_minus, eps_plus, p)
        eta, errors = estimate_errors(mesh, u, u_, f, 
                                      eps_minus, eps_plus, p)
        dofs = V.dim()
        cost += dofs

        # If we have exact solution available track difference in
        # energies
        error_energy = None
        if u_exact is not None:
            error_energy = (assemble(J_eps(u, f, p, eps_minus, eps_plus)) - 
                            assemble(J_eps(u_exact, f, p)))

        
        record = {"iter": iter_num, "n": n, 
                  "refinements": refinement_num,
                  "dofs": dofs, "cost": cost, 
                  "error_energy": error_energy,
                  "eps_minus": float(eps_minus), 
                  "eps_plus": float(eps_plus),
                  **{k: float(v) for k, v in errors.items()}}
        history.append(record)


        print(f"iter={iter_num} n={n} refs={refinement_num} "
              f"dofs={dofs} cost={cost} "
              f"eps=[{float(eps_minus):.3g},{float(eps_plus):.3g}] "
              + " ".join(f"{k}={v:.3g}" for k, v in errors.items())
              + (f" error_energy={error_energy:.3g}" 
              if error_energy is not None else ""))

        if (sum(errors.values()) <= tol):
            print(f"Converged at iter={iter_num}.")
            out_file.write(u, eta, time=iter_num)
            break

        ref_num_old = refinement_num
        n, refinement_num, max_refinement_reached = adapt(
                errors, eta, amh, eta_refinement, n,
                refinement_num, max_refinements,
                eps_minus, eps_plus
                )

        if max_refinement_reached:
            print(f"Warning!: discretization error" 
            f"({errors['disc']:.3g}) still dominates after "
            f"{max_refinements} refinements. Stopping.")
            out_file.write(u, eta, time=iter_num)
            break

        if (refinement_num != ref_num_old):
            out_file = VTKFile(
                       f"results/u_level_{refinement_num}.pvd"
                       )
            Q0 = FunctionSpace(mesh, "DG", 0)
            gradnorm = Function(Q0).interpolate(norm_safe(grad(u)))
            with gradnorm.dat.vec_ro as gv:
                print(f"  max|grad u| = {gv.norm(fd.PETSc.NormType.NORM_INFINITY):.3g}")

        iter_num += 1
        u_.assign(u)
        out_file.write(u, eta, time=iter_num)
    else:
         print(f"Warning!: Hit n_max={n_max} without converging "
               f"tolerance tol={tol}"
               )
    return u, history


# %%
# Set p to anything in between 1 and 2.
p = 16/10


# %%
# **Bump**
# Choose :math:`\Omega = [-1,1] \times [-1,1]` and choose
# :math:`f` and :math:`\text{Tr }u` such that:

# %%
# .. math::
#    u(x_1, x_2) = (x_1^2 - 1)(x_2^2 - 1)
def bump(p):
    """
    u(x1, x2) = (x1^2 - 1)(x2^2 - 1) on [-1,1]^2,
    u(x1, x2) = 0                    on boundary
    """
    def make_u_exact(mesh):
        x, y = SpatialCoordinate(mesh)
        u = (x**2 - 1) * (y**2 - 1)
        return u

    def make_f(mesh):
        x, y = SpatialCoordinate(mesh)
        u = (x**2 - 1) * (y**2 - 1)
        f = -div(norm_safe(grad(u))**(p-2) * grad(u))
        return f

    def make_bc(V):
        return DirichletBC(V, 0.0, "on_boundary")
    return make_u_exact, make_f, make_bc

# %%
# **Needle**
# On :math:`\Omega = [-1,1]^2` choose :math:`f` and 
# the boundary values of :math:`u` s.t. 
# :math:`u(x) = |x|^(1-1/p)`
def needle(p):
    """
    u(x) = |x|^{1-1/p} - 1 on [-1,1]^2.
    Singular gradient at the origin -- this is the paper's stress test
    for the estimator (V(grad u) has only borderline Lorentz-space regularity).
    """
    reg = Constant(0.05)   # regularizes the 0/0 at the origin; small enough
                            # not to visibly perturb u away from x=0

    def make_u_exact(mesh):
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x**2 + y**2 + reg**2)
        return r**(1 - 1/p) - 1

    def make_f(mesh):
        x, y = SpatialCoordinate(mesh)
        r = sqrt(x**2 + y**2 + reg**2)
        u = r**(1 - 1/p) - 1
        return -div(norm_safe(grad(u))**(p-2) * grad(u))

    def make_bc(V):
        x, y = SpatialCoordinate(V.mesh())
        r = sqrt(x**2 + y**2 + reg**2)
        u_bdry = r**(1 - 1/p) - 1
        return DirichletBC(V, u_bdry, "on_boundary")
    return make_u_exact, make_f, make_bc

# %% 
# Prepare the mesh :math:`\Omega = [-1,1] \times [-1,1]`
Omega = WorkPlane(Axes((0,0,0), n=Z, h=X)).RectangleC(2,2).Face()
geo = OCCGeometry(Omega, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.5)
mesh = Mesh(ngmsh)

# %%
# For the adaptive mesh refinment we will utilize firedrake's 
# :strong:`AdaptiveMeshHierarchy` and 
# :strong:`AdaptiveTransferManager` 
amh = AdaptiveMeshHierarchy(mesh)
atm = AdaptiveTransferManager()

# %% 
# Prepare boundary conditions, the rhs and the exact solution

# make_u_exact, make_f, make_bc = needle(p)
make_u_exact, make_f, make_bc = bump(p)

# %%
# Run the thing
u, history = p_poisson(p, make_f, make_bc,
                       make_u_exact, mesh,
                       amh, atm, tol=1e-8
                       )

# %%
# Function for plotting everything,
# once again this is Claude's work.
import matplotlib.pyplot as plt

def plot_convergence(history, case_name):
    cost = [h["cost"] for h in history]
    disc = [h["disc"] for h in history]
    lin  = [h["lin"] for h in history]
    plus = [h["plus"] for h in history]
    minus = [h["minus"] for h in history]
    energy = [h["error_energy"] for h in history if h["error_energy"] is not None]

    fig, ax = plt.subplots(figsize=(6, 5))
    if energy:
        ax.loglog(cost[:len(energy)], energy,
                  'k:', label=r"$J(u_n) - J(u)$")
    ax.loglog(cost, disc, label=r"$\eta_h^2$")
    ax.loglog(cost, lin, label=r"$\eta_{Kac}^2$")
    ax.loglog(cost, plus, label=r"$\eta_{\epsilon+}^2$")
    ax.loglog(cost, minus, label=r"$\eta_{\epsilon-}^2$")

    c0 = cost[0] if cost[0] > 0 else 1
    ref = [c0 / c for c in cost]   # reference slope: 1/cost
    ax.loglog(cost, ref, 'k-', linewidth=1,
              label=r"$\mathrm{cost}^{-1}$")

    ax.set_xlabel("accumulated cost")
    ax.set_ylabel("error")
    ax.set_title(case_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"results/{case_name}_convergence.pdf")
    plt.close(fig)

# %%
# Plot the errors vs acumulated cost
plot_convergence(history, "bump")
# plot_convergence(history, "needle")


# %%
# .. figure:: /_static/bump_convergence.png
#    :align: center

# %%
# Here is a plot of discretization error on levels 3, 6 and 15 for "**bump**" with :math:`p=16/10`.
#
# +----------------------------------+----------------------------------+------------------------------------+
# | .. figure:: /_static/eta_l3.png  | .. figure:: /_static/eta_l6.png  | .. figure:: /_static/eta_l15.png   |   
# |    :align: center                |    :align: center                |    :align: center                  |
# |    :height: 150px                |    :height: 150px                |    :height: 150px                  |
# |    :alt: Eta at level 3          |    :alt: Eta at level 6          |    :alt: Eta at level 15           |
# |                                  |                                  |                                    |
# |    *Level 3*                     |    *Level 6*                     |    *Level 15*                      |
# +----------------------------------+----------------------------------+------------------------------------+
#
# The solutions at level 4 and 15 are shown below.
#
# +----------------------------------------+---------------------------------------------+
# | .. figure:: /_static/solution_l4.png   | .. figure:: /_static/solution_l15.png       |    
# |    :align: center                      |    :align: center                           |
# |    :height: 300px                      |    :height: 300px                           |
# |    :alt: Solution, level 4             |    :alt: Solution, level 15                 |
# |                                        |                                             |
# |    *MG solution at level 4*            |    *MG solution at level 15*                |
# +----------------------------------------+---------------------------------------------+


# %% 
# A runnable version of this demo can be found :download:`here<relaxed_Kacanov.py>`.

# %%
# Bibliography
# ------------
# .. bibliography:: 
