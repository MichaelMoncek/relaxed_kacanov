"""
p-Poisson solver with adaptive error control in Firedrake
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

def kappa(t, eps_minus=0.0, eps_plus=float("inf"), p):
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

def J_eps(v, f, eps_minus=0.0, eps_plus=float("inf"), p):
    """
    Eq 2.2, Sec. 2 of (DFW22)
    """
    grad_norm = norm_safe(grad(v))
    return kappa(grad_norm, eps_minus, eps_plus, p)*dx - f*v*dx

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
    return (J_eps(v, f, eps_minus, eps_plus, p) -
            J_eps(v, f, eps_minus, float("inf"), p))

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
    return (J_eps(v, f, eps_minus, eps_plus, p) -
            J_eps(v, f, 0.0, eps_plus, p)) 

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
    eta = Function(Q).interpolate(sqrt(eta2))
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
# We will choose the later definition to work with.

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
# Implementation
# --------------

# %%
# Constructing the Domain
# -----------------------
# We first construct the domain over which we will solve the 
# problem.
# We begin with L-shaped domain, 
# build as a union of two rectangels: 

rect1 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(1,2).Face()
rect2 = WorkPlane(Axes((0,0,0), n=Z, h=X)).Rectangle(2,1).Face()
# L = rect1 + rect2
L = rect1 

geo = OCCGeometry(L, dim=2)
ngmsh = geo.GenerateMesh(maxh=0.5)
mesh = Mesh(ngmsh)

# %%
# For the adaptive mesh refinment we will utilize firedrake's 
# :strong:`AdaptiveMeshHierarchy` and 
# :strong:`AdaptiveTransferManager` 

amh = AdaptiveMeshHierarchy(mesh)
atm = AdaptiveTransferManager()

# %%
# Do one Kacanov step
def kacanov_step(u_, f, bc, V, atm, eps_minus, eps_plus, p):
    """
    Do one Kacanov step i.e. Solve the linear poisson problem
    -div(a*grad(u)) = f with coefficient matrix
    a = clamp(grad(u_), eps_minus, eps_plus)
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    bc = fd.DirichletBC(V, 0.0, "on_boundary")

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
# Return the a posteriori error estimates
def estimate_errors(mesh, u, u_, f, eps_minus, eps_plus, p):
    """
    Returns discretization error on each element and dictionary 
    of linearization, plus, minus and discretization error size
    """
    error2_minus = assemble(eta2_minus(u, f, eps_minus, eps_plus, p))
    error2_plus  = assemble(eta2_plus(u, f, eps_minus, eps_plus, p))
    error2_lin   = assemble(eta2_lin(u, u_, eps_minus, eps_plus, p))
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
    # if sum(errors.values()) <= tol:
    #    break

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
       if (refinement_num <= max_refinements):
           amh.adapt(eta, eta_refinement)
           refinement_num += 1
       else:
           n += 1
    return n

# %%
# p-Poisson solver function
def p_poisson(p, mesh, amh, atm, tol=1e-8):
    """
    The main loop 
    """
    p = Constant(p)
    eps_plus  = Constant(1.0)
    eps_minus = Constant(1.0)
    eps_tol   = 10.0**6
    eta_refinement = 0.5
    n, n_max  = 0, 1000
    u_ = None

    out_file = fd.VTKFile("results/un_level_0.pvd")
    while (n < n_max):
        mesh = amh[-1]
        # set up the problem
        V = FunctionSpace(mesh, 'Lagrange', 1)
        x, y = SpatialCoordinate(mesh)
        f = Function(V).interpolate(Constant(1.0))
        bc = DirichletBC(V, 0.0, "on_boundary")

        if u_ is None:
           u_ = Function(V)
        elif u_.function_space() != V:
           u_new = Function(V)
           atm.prolong(u_, u_new)
           u_ = u_new

        u = kacanov_step(u_, f, bc, V, atm, eps_minus, eps_plus, p)
        eta, errors = estimate_errors(mesh, u, u_, f, 
                                      eps_minus, eps_plus, p)
        n = adapt(errors, eta, amh, eta_refinement, n,
              refinement_num, max_refinements,
              eps_minus, eps_plus, eps_tol=1e6):

        u_.assign(u)
        out_file.write(u)
        out_file.write(eta)


# %%
# def p_laplace(p, mesh, tol=10e-8):
#     p = fd.Constant(p) 
#
#     out_file = fd.VTKFile("results/un_level_0.pvd")
#
#     eps_plus = fd.Constant(1.0)
#     eps_minus = fd.Constant(1.0)
#     eps_tol = 10.0**16
#
#     eta_refinement = 0.5
#     n = 0
#     n_max = 1000
#     iter = 0
#     level_iter = 0
#     refinement_num = 0
#     max_refinements = 10
#     u_ = None
#
#
#     while (n < n_max): # main loop
#        mesh = amh[-1]
#
#        x, y = fd.SpatialCoordinate(mesh)
#
#        V = fd.FunctionSpace(mesh, 'Lagrange', 1)
#        # f = 1.0 + cos(2.0*pi*x)*sin(2.0*pi*y)
#        f = Function(V).interpolate(Constant(1.0))
#
#        u = fd.TrialFunction(V)
#        v = fd.TestFunction(V)
#        un = fd.Function(V, name="un")
#        if u_ is None:
#            u_ = Function(V)
#        elif u_.function_space() != V:
#            u_new = fd.Function(V)
#            atm.prolong(u_, u_new)
#            u_ = u_new
#
#        bc = fd.DirichletBC(V, 0.0, "on_boundary")
#
#        grad_norm = norm_safe(grad(u_))
#        a = clamp(grad_norm, eps_minus, eps_plus)**(p-2)
#
#        F = a*inner(grad(u), grad(v))*dx - f*v*dx 
#        problem = fd.LinearVariationalProblem(fd.lhs(F), fd.rhs(F),
#                                              un, bc)
#        sp = {"ksp_type": "preonly", "pc_type": "lu",
#              "pc_factor_mat_solver_type": "mumps"}
#        solver = fd.LinearVariationalSolver(problem,
#                                            solver_parameters=sp)
#
#        solver.set_transfer_manager(atm)
#        solver.solve()
#
#        # a posteriori estimates
#        error2_minus = (fd.assemble(eta2_minus(un, f, 
#                                             eps_minus, eps_plus, p)))
#        error2_plus= fd.assemble(eta2_plus(un, f, 
#                                          eps_minus, eps_plus, p))
#        error2_lin = fd.assemble(eta2_lin(un, u_, 
#                                              eps_minus, eps_plus, p))
#
#        eta, error2_disc = eta2_disc(mesh, un, f, 
#                                      eps_minus, eps_plus, p)
#
#        errors = {"lin" : error2_lin,
#                  "plus" : error2_plus,
#                  "minus" : error2_minus,
#                  "disc" : error2_disc
#                  }
#
#        if sum(errors.values()) <= tol:
#            break
#
#        largest_error = max(errors, key=errors.get)
#        if largest_error == "lin":
#            n += 1
#        elif largest_error == "plus":
#            if (abs(float(eps_plus)) <= eps_tol):
#                eps_plus.assign(1.25*eps_plus)
#            else:
#                n += 1
#        elif largest_error == "minus":
#            if (abs(float(eps_minus)) >= 1/eps_tol):
#                eps_minus.assign(0.80*eps_minus)
#            else:
#                n += 1
#        else: # "disc"
#            if (refinement_num <= max_refinements):
#                amh.adapt(eta, eta_refinement)
#                refinement_num += 1
#            else:
#                n += 1
#            level_iter = 0
#            out_file = fd.VTKFile(f"results/un_level_{refinement_num}.pvd")
#
#        print(f"n = {n}, iteration = {iter}, refinements = {refinement_num}")
#        print(f"Discretization Error is: {float(error2_disc):.3g}")
#        print(f"Linearization Error is: {float(error2_lin):.3g}")
#        print(f"Epsilon is [{float(eps_minus):.3g}, {float(eps_plus):.3g}]")
#        print(f"Error_plus is: {float(error2_plus):.3g}")
#        print(f"Error_minus is: {float(error2_minus):.3g}")
#        # print(float(eps_minus), float(eps_plus))
#        #
#        # E = fd.assemble(energy(u_, f, p, eps_minus, eps_plus))
#        # E_minus = fd.assemble(energy(u_, f, p, eps_plus=eps_plus))
#        #
#        # print("E       =", E)
#        # print("E_minus =", E_minus)
#        # print("error   =", E - E_minus)
#        u_.assign(un) 
#
#        iter += 1
#        level_iter += 1
#
#        out_file.write(un, time=level_iter)

# problem parameters
# p = 16/15
p = 16/15
# p_laplace(p, mesh, tol=1e-12)


# %%
# Bibliography
# ------------
# .. bibliography:: 
