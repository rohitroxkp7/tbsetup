from baseclasses import AeroProblem
from . import constants

def setup(args):

    # Geometry constants
    chordRef = 3.25#constants.geometry[args.geometry]["chord"]
    areaRef = 45.5#chordRef * 1.0 # 1 meter / 1 cell wide

    # Reference and Rotation center
    xRef = 0.25
    xRot = 0.25

    # Gas constants
    gamma = 1.4
    R = 287.085

    # Set flow condition based on the case
    reynolds=6.5e6
    T =300 # 300 is the average of the tunnel temp range

    M=args.mach
    alpha=args.alpha

    name = "OAT15A"

    # Aerodynamic problem description
    ap = AeroProblem(name=name, alpha=alpha,  mach=M, machRef=M,
                    areaRef=areaRef, chordRef=chordRef,
                    reynolds=reynolds,reynoldsLength=chordRef,
                    evalFuncs=["cl","cd","cmz"],
                    gamma=gamma, T=T, R=R,
                    xRef=xRef, yRef=0.0, zRef=0.0,
                    xRot=xRot, yRot=0.0, zRot=0.0)

    return ap

