# ======================================================================
#         Import modules
# ======================================================================

import os
import argparse
import shutil
import inspect
from pprint import pprint as pp
import time

import numpy
from mpi4py import MPI
from baseclasses import *
#from multipoint import redirectIO
from adflow import ADFLOW
from baseclasses.utils import redirectIO

from SETUP import setup_problem

comm = MPI.COMM_WORLD

# ======================================================================
#         Input Information
# ======================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="debug", help="This is the solution directory.")
parser.add_argument("--outDir", type=str, default="debug", help="This is the base output directory. Run case directories are then created here.")
parser.add_argument("--geometry", choices=["G1", "G2"], default="G2")
parser.add_argument("--gridFamily", choices=["A", "B"], default="A")
parser.add_argument("--gridLevel", choices=["L0", "L1", "L2"], default="L0")
parser.add_argument("--smoother", choices=["DADI", "Runge-Kutta"], default="DADI" , help="[default: %(default)s]")
parser.add_argument("--useNK", action="store_true", default=False, help="[default: %(default)s]")
parser.add_argument("--useANK", action="store_true", default=False, help="[default: %(default)s]")
parser.add_argument("--mgCycle", type=str, default="3w", help="Set the level of multigrid [default: %(default)s]")
parser.add_argument("--nCycles", type=int, default=50000, help="Set the number of file cycles done for fine grid [default: %(default)s]")
parser.add_argument("--fineCFL", type=float, default=1.5, help="Set the CFL number for fine mesh [default: %(default)s]")
parser.add_argument("--coarseCFL", type=float, default=1.5, help="Set the CFL number for the coarse mesh [default: %(default)s]")
parser.add_argument("--turbModel", choices=["SA"], default="SA", help="[default: %(default)s]")
parser.add_argument("--discretization", choices=["central plus scalar dissipation", "central plus matrix dissipation", "upwind"], default="central plus scalar dissipation", help="[default: %(default)s]")
parser.add_argument("--nPeriods", type=int, default=4, help="[Number of periods to run. default: %(default)s]")
parser.add_argument("--nStepPerPeriod", type=int, default=64, help="[Number of timesteps per period. default: %(default)s]")
parser.add_argument("--mach", type=float, default=0.734, help="[default: %(default)s]")
parser.add_argument("--alpha", type=float, default=4.8, help="[default: %(default)s]")
parser.add_argument("--vis2", type=float, default=0.25, help="[default: %(default)s]")
parser.add_argument("--vis4", type=float, default=0.05, help="[default: %(default)s]")
args = parser.parse_args()  


# Reference all files in a dictionary so we can auto-save them
files = {}
files["self"] = inspect.getfile(inspect.currentframe())
files["gridFile"] = f"./mesh/RAE2822.cgns"#OAT15A_{args.geometry}_{args.gridFamily}_{args.gridLevel}.cgns"

# Set output base directory
outDir = os.path.join(args.outDir, f"m{args.mach}_a{args.alpha}", args.output)
solDir = os.path.join(outDir, "output")

# Make backup of what was actually run
if comm.rank == 0:
    # Create a folder called input in the output directory to save all input files.
    copyDir = os.path.join(outDir, "INPUT")
    os.system(f"mkdir -p {copyDir}")
    for key in files:
        shutil.copy(files[key], copyDir)

    # Create a folder of the output files
    os.system(f"mkdir -p {solDir}")

comm.barrier()

#Redirect STDOUT
if comm.rank == 0:
    #fName = os.path.join(outDir, "%s_%d.out"%(setName, ptID))
    fName = os.path.join(outDir, "std.out")
    outFile = open(fName, "w")
    #redirectIO(outFile)
    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)

        def flush(self):
            for f in self.files:
                f.flush()

    #sys.stdout = Tee(sys.stdout, outFile)
    #sys.stdout = outFile


# ======================================================================
#         Setup problem
# ======================================================================
ap = setup_problem.setup(args)

# Unsteady parameter computation
# Use the reduced frequency to find the angular velocity
#omega = 2*V*k/c = 2*M*a*k/c = 2*M*sqrt(gamma*R*T)*k/c
#print k, M, R, T, chordRef, gamma


nfineSteps = 2000
dt = 5e-4 # [s] The actual timestep
T0 = dt * nfineSteps # [s] Total time

# Make dictionary for ease of printing to terminal
timeAnalysisOptions = {
    "nfineSteps" : nfineSteps,
    "dt": dt,
    "T0": T0,
    }

aeroOptions = {
    # Common Parameters
    "gridFile":files["gridFile"],
    "outputDirectory":solDir,
    "liftindex":2,
    "writevolumesolution":True,
    "writesurfacesolution":False,
    "writeTecplotSurfaceSolution":False,
    "surfaceVariables": ["soundspeed","p","rho","cp","vx","vy","vz"],
    #"surfacevariables":["cp","vx","vy","vz","mach","rvx"],
    "rkreset":True,
    "nrkreset":10,

    # Physics Parameters
    "equationType":"RANS",
    "equationMode":"unsteady",
    "timeIntegrationScheme":"BDF",
    "nTimeStepsFine":nfineSteps,
    "deltaT":dt,

    "discretization": args.discretization,
    "coarsediscretization": args.discretization,
    "vis2": args.vis2,
    "vis4": args.vis4,


    "smoother":args.smoother,
#    "nsubiterturb":3, #10
#    "nsubiter":5, #5
    "turbulenceModel":args.turbModel,
    #"turbulenceProduction": "strain",
    "useft2SA": False,
    "useRotationSA": False,
    # Grid parameters
    "useALE":True,
    "useGridMotion": True,


    # Common Parameters
    "CFL": args.fineCFL,
    "CFLCoarse": args.coarseCFL,
    "MGCycle": args.mgCycle,
    "MGStartLevel":1, # -1 FIX NEEDED
    "nCycles":args.nCycles,
    "ncyclescoarse":50000,
    #"monitorvariables":["cpu", "resrho","resturb","cl","cd", "yplus","totalr"],
    #"monitorvariables":["cpu", "resrho","cl","cd", "yplus","totalr"],
    "monitorvariables":["cpu","resrho", "resTurb", "cl", "cd", "cmz"],
    "isosurface":{"shock":1.0, "vx":-0.001},
    "useNKSolver":args.useNK,
    "useANKSolver":args.useANK,
    #"writeSolutionEachIter": True,

    # Convergence Parameters
    "L2Convergence":1e-1,
    "L2ConvergenceCoarse":1e-1,

    # NK Solver parameters
    "nkswitchtol":5e-5,
    "nkouterpreconits":3,
    "nksubspacesize":80,
    "nkjacobianlag":5,
    "nkswitchtol":1e-4,
    "nkadpc":True,
#    "nkls":"non monotone",
    #"restartFile":["/home/rohit/Desktop/Adflow_transonic_buffet/buffet/output/m0.73_a4.0/debug/output/vol1.cgns","/home/rohit/Desktop/Adflow_transonic_buffet/buffet/output/m0.73_a4.0/debug/output/vol2.cgns"]
    }


meshOptions = {
    "gridFile":files["gridFile"],
    }

# Echo the various options:
if comm.rank == 0:
    print("This script was executed: ", time.strftime("%Y-%m-%d %H:%M"))

    print("+------------------------------------------------+")
    print("+            Command Line Options                +")
    print("+------------------------------------------------+")
    pp(vars(args))

    print("+------------------------------------------------+")
    print("|            AeroProblem Options                 |")
    print("+------------------------------------------------+")
    pp(vars(ap))

    print("+------------------------------------------------+")
    print("|            Frequency and analysis options      |")
    print("+------------------------------------------------+")
    pp(timeAnalysisOptions)


# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

# Add slices
CFDSolver.addSlices("z",[0.5])
CFDSolver.addLiftDistribution(2,"z")
CFDSolver(ap)

# Evaluate functions
funcs = {}
CFDSolver.evalFunctions(ap, funcs)

# Print the evaluated functions
if comm.rank == 0:
    print(funcs)


