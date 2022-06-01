# Navier-Stokes equations
# =======================
#
# We solve the Navier-Stokes equations using Taylor-Hood elements. 
# Solvers are compared : direct mumps, IT solver (hypre) and MG. The
# example is that of a lid-driven cavity,original file : https://www.firedrakeproject.org/demos/navier_stokes.py 

from firedrake import *
import time

N = 4
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


# Mumps direct solver tuned
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
if(Z.dim()<Max_dofs/4.0):
    solve(F == 0, up, bcs=bcs, nullspace=nullspace, solver_parameters=parameters,
      appctx=appctx)
final=time.time()
print("Time for the iterative solver : %s" %(final-initial))
class Mass(AuxiliaryOperatorPC):

    def form(self, pc, test, trial):
        a = 1/nu * inner(test, trial)*dx
        bcs = None
        return (a, bcs)
# Case 3 : test the MG solver
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
