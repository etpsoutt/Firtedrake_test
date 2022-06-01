# A saddle-point system: The Stokes equations
# -------------------------------------------
# Original file : https://www.firedrakeproject.org/demos/geometric_multigrid.py.html
# Having demonstrated basic usage, we'll now move on to an example where
# the configuration of the multigrid solver is somewhat more complex.
# This demonstrates how the multigrid functionality composes with the
# other aspects of solver configuration, like fieldsplit
# preconditioning.  We'll use Taylor-Hood elements and solve a problem
# with specified velocity inflow and outflow conditions. 
# Update : we compare with different meshes the performance of three (parallel) solvers : 
# 1. Mumps direct solver
# 2. The first MG-solver proposed in firedrake
# 3. fgmres hypre-ilu solver
# 4. A second MG-solver proposed in firedrake
# to run in parallel use command line : OMP_NUM_THREADS=1 mpiexec -np 8 python3 stokes_mgtuto.py
from firedrake import *
import time
mesh = RectangleMesh(15, 10, 1.5, 1)
N=4
Max_dofs=6000000 # On my computed, only the MG solvers were able to compute a parallel solution with more than 6e6 DOFs
hierarchy = MeshHierarchy(mesh, N)

mesh = hierarchy[-1]

V = VectorFunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 1)
Z = V * W
print("Number of degrees of freedom : %s" %Z.dim())

u, p = TrialFunctions(Z)
v, q = TestFunctions(Z)
nu = Constant(1)

a = (nu*inner(grad(u), grad(v)) - p * div(v) + div(u) * q)*dx

L = inner(Constant((0, 0)), v) * dx

x, y = SpatialCoordinate(mesh)

t = conditional(y < 0.5, y - 0.25, y - 0.75)
l = 1.0/6.0
gbar = conditional(Or(And(0.25 - l/2 < y,
y < 0.25 + l/2),
And(0.75 - l/2 < y,
y < 0.75 + l/2)),
Constant(1.0), Constant(0.0))

value = gbar*(1 - (2*t/l)**2)
inflowoutflow = Function(V).interpolate(as_vector([value, 0]))
bcs = [DirichletBC(Z.sub(0), inflowoutflow, (1)),
DirichletBC(Z.sub(0), zero(2), (3, 4))]

# First up, we'll use an algebraic preconditioner, with a direct solve,
# remembering to tell PETSc to use pivoting in the factorisation. ::

udirect = Function(Z)
parameters = {
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_type":"newtonls",
    "snes_view": None,
    "snes_monitor": None,
    "ksp_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1,
    "mat_mumps_icntl_2": None,
}
# only solve if the number is small enough otherwise an out of memory crash will occur
if(Z.dim()<Max_dofs):
    initial=time.time()
    solve(a == L, udirect, bcs=bcs, solver_parameters=parameters)
    final=time.time()
    print("Time resolution for the direct solver was %s" %(final-initial))
                                             

# Next we'll use a Schur complement solver, using geometric multigrid to
# invert the velocity block. The Schur complement is spectrally equivalent
# to the viscosity-weighted pressure mass matrix. Since the pressure mass
# matrix does not appear in the original form, we need to supply its
# bilinear form to the solver ourselves: ::

class Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = 1/nu * inner(test, trial)*dx
        bcs = None
        return (a, bcs)

parameters = {
    "ksp_type": "gmres",
    "ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "mg",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "__main__.Mass",
    "fieldsplit_1_aux_pc_type": "bjacobi",
    "fieldsplit_1_aux_sub_pc_type": "icc",
}

umg1 = Function(Z)
initial=time.time()
solve(a == L, umg1, bcs=bcs, solver_parameters=parameters)
final=time.time()
print("Time resolution for the mg1 solver was %s" %(final-initial))


# Try here the most naive approach : a flexible gmres algorithm
parameters= {
    "snes_rtol": 1e-5,
    "snes_atol": 1e-8,
    #"snes_monitor": None,
    #"snes_view": None,
    "snes_converged_reason": None,
    "snes_type":"newtonls",
    #"snes_type": "Newton linesearch",
    "ksp_type": "fgmres",
    #"ksp_monitor": None,
    #"ksp_view": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-50,
    "ksp_divtol": 1e4,
    "ksp_rtol":1e-3,
    "ksp_max_it": 10000,
    "mat_type": "aij",
    "pc_type": "hypre",
    "pc_hypre_type":"euclid",
    "pc_hypre_euclid_levels": 1,
    #"pc_type": "ilu",
    #"pc_factor_levels":1,
    #"pc_factor_reuse_ordering":None,
    #"pc_factor_reuse_fill":None,
    "ksp_gmres_restart": 150,
}
uiter1 = Function(Z)
initial=time.time()
if(Z.dim()<Max_dofs):
    solve(a == L, uiter1, bcs=bcs, solver_parameters=parameters)
final=time.time()
print("Time resolution for the iterative solver was %s" %(final-initial))

# Finally, we'll use coupled geometric multigrid on the full problem,
# using Schur complement "smoothers" on each level. On the coarse grid
# we use a full factorisation for the velocity and Schur complement
# approximations, whereas on the finer levels we use incomplete
# factorisations for the velocity block and Schur complement
# approximations.
#
# .. note::
#
#    If we wanted to just use LU for the velocity-pressure system on the
#    coarse grid we would have to say ``"mat_type": "aij"``, rather than
#    ``"mat_type": "nest"``.
#
# ::

parameters = {
      "ksp_type": "gcr",
      "ksp_monitor": None,
      "mat_type": "nest",
      "pc_type": "mg",
      "ksp_rtol": 1e-8,
      "mg_coarse_ksp_type": "preonly",
      "mg_coarse_pc_type": "fieldsplit",
      "mg_coarse_pc_fieldsplit_type": "schur",
      "mg_coarse_pc_fieldsplit_schur_fact_type": "full",
      "mg_coarse_fieldsplit_0_ksp_type": "preonly",
      "mg_coarse_fieldsplit_0_pc_type": "lu",
      "mg_coarse_fieldsplit_1_ksp_type": "preonly",
      "mg_coarse_fieldsplit_1_pc_type": "python",
      "mg_coarse_fieldsplit_1_pc_python_type": "__main__.Mass",
      "mg_coarse_fieldsplit_1_aux_pc_type": "cholesky",
      "mg_levels_ksp_type": "richardson",
      "mg_levels_ksp_max_it": 1,
      "mg_levels_pc_type": "fieldsplit",
      "mg_levels_pc_fieldsplit_type": "schur",
      "mg_levels_pc_fieldsplit_schur_fact_type": "upper",
      "mg_levels_fieldsplit_0_ksp_type": "richardson",
      "mg_levels_fieldsplit_0_ksp_convergence_test": "skip",
      "mg_levels_fieldsplit_0_ksp_max_it": 2,
      "mg_levels_fieldsplit_0_ksp_richardson_self_scale": None,
      "mg_levels_fieldsplit_0_pc_type": "bjacobi",
      "mg_levels_fieldsplit_0_sub_pc_type": "ilu",
      "mg_levels_fieldsplit_1_ksp_type": "richardson",
      "mg_levels_fieldsplit_1_ksp_convergence_test": "skip",
      "mg_levels_fieldsplit_1_ksp_richardson_self_scale": None,
      "mg_levels_fieldsplit_1_ksp_max_it": 3,
      "mg_levels_fieldsplit_1_pc_type": "python",
      "mg_levels_fieldsplit_1_pc_python_type": "__main__.Mass",
      "mg_levels_fieldsplit_1_aux_pc_type": "bjacobi",
      "mg_levels_fieldsplit_1_aux_sub_pc_type": "icc",
}


u = Function(Z)
initial=time.time()
solve(a == L, u, bcs=bcs, solver_parameters=parameters)
final=time.time()
print("Time resolution for the mg2 solver was %s" %(final-initial))
# Finally, we'll write the solution for visualisation with Paraview. ::
if(Z.dim()<Max_dofs):
    ud,pd=udirect.split()
    u1,p1=umg1.split()
    u2,p2=u.split()
    u3,p3=uiter1.split()
    print("Error of mg1 with DS for the velocity: %s" % errornorm(ud,u1))
    print("Error of mg1 with DS for the pressure: %s" % errornorm(pd,p1))
    print("Error of mg2 with DS for the velocity: %s" % errornorm(ud,u2))
    print("Error of mg2 with DS for the pressure: %s" % errornorm(pd,p2))
    print("Error of iter scheme with DS for the velocity: %s" % errornorm(ud,u3))
    print("Error of iter scheme with DS for the pressure: %s" % errornorm(pd,p3))

u, p = u.split()
u.rename("Velocity")
p.rename("Pressure")

File("stokes.pvd").write(u, p)

