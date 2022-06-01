# Navier-Stokes equations
# =======================
#
# We solve the Navier-Stokes equations using Taylor-Hood elements. 
# Solvers are compared : direct mumps, IT solver (hypre) and MG. The
# example is that of a lid-driven cavity,original file : https://www.firedrakeproject.org/demos/navier_stokes.py ::

from firedrake import *
import time

N = 6
Max_dofs=3000000 # More than 2 mios of dofs result in out of memory error for my computer with direct solver
initial=time.time()
Mesh = UnitSquareMesh(8, 8)
hierarchy = MeshHierarchy(Mesh, N)

M = hierarchy[-1]
nu = Constant(1)

V = VectorFunctionSpace(M, "CG", 2,name="V")
W = FunctionSpace(M, "CG", 1,name="P")
Z = V * W
print("Number of degrees of freedom : %s" %Z.dim())

up = Function(Z)
u, p = split(up)
v, q = TestFunctions(Z)

Re = Constant(100.0)

F = (
    1.0 / Re * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nullspace = MixedVectorSpaceBasis(
    Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

# Having set up the problem, we now move on to solving it.  Some
# preconditioners, for example pressure convection-diffusion (PCD), require
# information about the the problem that is not easily accessible from
# the bilinear form.  In the case of PCD, we need the Reynolds number
# and additionally which part of the mixed velocity-pressure space the
# velocity corresponds to.  We provide this information to
# preconditioners by passing in a dictionary context to the solver.
# This is propagated down through the matrix-free operators and is
# therefore accessible to custom preconditioners. ::

appctx = {"Re": Re, "velocity_space": 0}

# Now we'll solve the problem.  First, using a direct solver.  Again, if
# MUMPS is not installed, this solve will not work, so we wrap the solve
# in a ``try/except`` block. ::
# Case 1 : test the mumps settings

from firedrake.petsc import PETSc
initial=time.time()
try:
    if(Z.dim()<Max_dofs/4.0):
        solve(F == 0, up, bcs=bcs, nullspace=nullspace,
          solver_parameters={"snes_monitor": None,
                             "ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"})
except PETSc.Error as e:
    if e.ierr == 92:
        warning("MUMPS not installed, skipping direct solve")
    else:
        raise e
final=time.time()
print("Time for the direct solver : %s" %(final-initial))
# Now we'll show an example using the :class:`~.PCDPC` preconditioner
# that implements the pressure convection-diffusion approximation to the
# pressure Schur complement.  We'll need more solver parameters this
# time, so again we'll set those up in a dictionary. ::

parameters = {"mat_type": "matfree",
              "snes_monitor": None,

# We'll use a non-stationary Krylov solve for the Schur complement, so
# we need to use a flexible Krylov method on the outside. ::

             "ksp_type": "fgmres",
             "ksp_gmres_modifiedgramschmidt": None,
             "ksp_monitor_true_residual": None,

# Now to configure the preconditioner::

             "pc_type": "fieldsplit",
             "pc_fieldsplit_type": "schur",
             "pc_fieldsplit_schur_fact_type": "lower",

# we invert the velocity block with LU::

             "fieldsplit_0_ksp_type": "preonly",
             "fieldsplit_0_pc_type": "python",
             "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
             "fieldsplit_0_assembled_pc_type": "lu",

# and invert the schur complement inexactly using GMRES, preconditioned
# with PCD. ::

             "fieldsplit_1_ksp_type": "gmres",
             "fieldsplit_1_ksp_rtol": 1e-4,
             "fieldsplit_1_pc_type": "python",
             "fieldsplit_1_pc_python_type": "firedrake.PCDPC",

# We now need to configure the mass and stiffness solvers in the PCD
# preconditioner.  For this example, we will just invert them with LU,
# although of course we can use a scalable method if we wish. First the
# mass solve::

             "fieldsplit_1_pcd_Mp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Mp_pc_type": "lu",

# and the stiffness solve.::

             "fieldsplit_1_pcd_Kp_ksp_type": "preonly",
             "fieldsplit_1_pcd_Kp_pc_type": "lu",

# Finally, we just need to decide whether to apply the action of the
# pressure-space convection-diffusion operator with an assembled matrix
# or matrix free.  Here we will use matrix-free::

# Emile : I prefer to use a slightly tuned direct mumps solver instead

             "fieldsplit_1_pcd_Fp_mat_type": "matfree"}
parameters = {
    "snes_rtol": 1e-8,
    "snes_atol": 1e-8,
    "snes_type":"newtonls",
    #"snes_view": None,
    "snes_monitor": None,
    #"ksp_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "preonly",
    "mat_type": "aij",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 200,
    "mat_mumps_icntl_24": 1,
    "mat_mumps_icntl_2": None,
}

# With the parameters set up, we can solve the problem, remembering to
# pass in the application context so that the PCD preconditioner can
# find the Reynolds number. ::
# Test other parameters 
u1=Function(Z)
u1.assign(up)
up.assign(0)
initial=time.time()
# only solve Direct solver if the DOFS is low enough
if(Z.dim()<Max_dofs):
    solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)
final=time.time()
print("Time for the direct mumps solver : %s" %(final-initial))
# Case 2 : test the standard fgmres settings
# SAME BEHAVIOUR FOUND HERE
parameters = {
    "snes_rtol": 1e-4,
    "snes_atol": 1e-8,
    "snes_monitor": None,
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
    "pc_type": "ilu",
    "pc_factor_levels":1,
    "pc_factor_reuse_ordering":None,
    "pc_factor_reuse_fill":None,
    #"pc_factor_nonzeros_along_diagonal":None,# This parameter is in conflict with nullspace!
    "pc_factor_pivot_in_blocks":None,
    "ksp_gmres_restart": 150,
}
parameters = {
    "snes_rtol": 1e-6,
    "snes_atol": 1e-8,
    "snes_monitor": None,
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
    "ksp_gmres_restart": 150,
}
u2=Function(Z)
u2.assign(up)
up.assign(0)
initial=time.time()
if(Z.dim()<Max_dofs):
    solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)
final=time.time()
print("Time for the iterative solver : %s" %(final-initial))
parameters = {
    "snes_rtol": 1e-4,
    "snes_atol": 1e-8,
    "snes_monitor": None,
    "snes_view": None,
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
    "pc_type": "ilu",
    "pc_factor_levels":1,
    "pc_factor_reuse_ordering":None,
    "pc_factor_reuse_fill":None,
    "pc_factor_pivot_in_blocks":None,
    "ksp_gmres_restart": 150,
}
# if running in parallel use hypre and not ilu
parameters = {
    "snes_rtol": 1e-4,
    "snes_atol": 1e-8,
    "snes_monitor": None,
    "snes_view": None,
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
    "ksp_gmres_restart": 150,
}
class Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = 1/nu * inner(test, trial)*dx
        bcs = None
        return (a, bcs)
# Case 3 : test the schur split coupled to optimised fgmres this method is not compatible with nullspace (probably?)
"""parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_monitor": None,
    "snes_rtol": 1.0e-4,
    "snes_atol": 1.0e-4,
    "snes_max_linear_solve_fail": 10,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "mat_type": "aij",
    "ksp_rtol": 1.0e-4,
    "ksp_atol": 1.0e-4,
    "ksp_max_it": 2000,
    "ksp_gmres_restart": 150,
    "ksp_monitor": None,
    "ksp_converged_reason": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "selfp",
    "pc_fieldsplit_V_fields": "0,1",
    "pc_fieldsplit_P_fields": "2",
    "fieldsplit_V_fieldsplit_0_pc_type": "ilu",
    "fieldsplit_V_fieldsplit_1_pc_type": "ilu",
    "fieldsplit_V_fieldsplit_0_pc_factor_levels":1,
    "fieldsplit_V_fieldsplit_1_pc_factor_levels":1,
    "fieldsplit_V_fieldsplit_0_pc_factor_reuse_ordering":None,
    "fieldsplit_V_fieldsplit_0_pc_factor_reuse_fill":None,
    "fieldsplit_V_fieldsplit_0_pc_factor_pivot_in_blocks":None,
    #"fieldsplit_V_fieldsplit_0_pc_fieldsplit_schur_fact_type": "upper",
    "fieldsplit_V_fieldsplit_1_pc_factor_reuse_ordering":None,
    "fieldsplit_V_fieldsplit_1_pc_factor_reuse_fill":None,
    "fieldsplit_V_fieldsplit_1_pc_factor_pivot_in_blocks":None,
    #"fieldsplit_V_fieldsplit_1_pc_fieldsplit_schur_fact_type": "upper",
    "fieldsplit_P_pc_type": "ilu",
    "fieldsplit_P_pc_factor_levels":3,
    "fieldsplit_P_pc_factor_reuse_ordering":None,
    "fieldsplit_P_pc_factor_reuse_fill":None,
    "fieldsplit_P_pc_factor_pivot_in_blocks":None,
}"""
parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_monitor": None,
    "snes_rtol": 1.0e-6,
    "snes_atol": 1.0e-4,
    "snes_max_linear_solve_fail": 10,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "mat_type": "aij",
    #"ksp_type": "gmres",
    #"ksp_monitor": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "fieldsplit_V_ksp_type": "preonly",
    "fieldsplit_V_pc_type": "mg",
    "fieldsplit_P_ksp_type": "preonly",
    "fieldsplit_P_pc_type": "python",
    "fieldsplit_P_pc_python_type": "__main__.Mass",
    "fieldsplit_P_aux_pc_type": "bjacobi",
    "fieldsplit_P_aux_sub_pc_type": "icc",
    }
u3=Function(Z)
u3.assign(up)
up.assign(0)
initial=time.time()
solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)
final=time.time()
print("Time for the  MG solver : %s" %(final-initial))

# solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
#     appctx=appctx)

# And finally we write the results to a file for visualisation. ::

uv1,p1 = up.split()
u,p=u2.split()
u.rename("Velocity")
p.rename("Pressure")
if(Z.dim()<Max_dofs/4.0):
    uv2,p2=u.split()
uv3,p3=u3.split()
if(Z.dim()<Max_dofs/4.0):
    print("Error of fieldsplit with DS for the velocity: %s" % errornorm(u,uv2))
    print("Error of fieldsplit with DS for the pressure: %s" % errornorm(p,p2))
if(Z.dim()<Max_dofs):
    print("Error of iterative with DS for the velocity: %s" % errornorm(u,uv3))
    print("Error of iterative with DS for the pressure: %s" % errornorm(p,p3))
    print("Error of MG scheme with DS for the velocity: %s" % errornorm(u,uv1))
    print("Error of MG scheme with DS for the pressure: %s" % errornorm(p,p1))

File("cavity.pvd").write(u, p)
final=time.time()
# print("Time of resolution of the system in seconds : %s" % (final-initial))

# A runnable python script implementing this demo file is available
# `here <navier_stokes.py>`__.