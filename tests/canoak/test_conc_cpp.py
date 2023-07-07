import canoak
import numpy as np

sze3 = 152
jtot = 30
jtot3 = 150
met_zl = 1.5
delz = 0.5
izref = 1
cref = 2.2
soilflux = 5.4
factor = 4.5
ustar_ref = 0.5
ustar = 1.8

source = np.ones(jtot, dtype="float64")
cncc = np.ones(jtot3, dtype="float64") + 0.2
dispersion_np = np.ones([jtot3, jtot], dtype="float64") + 0.5

print(type(source[0]))
print(cncc)

canoak.conc(  # type: ignore
    cref,
    soilflux,
    factor,
    sze3,
    jtot,
    jtot3,
    met_zl,
    delz,
    izref,
    ustar_ref,
    ustar,
    source,
    cncc,
    dispersion_np
    # cref=cref, soilflux=soilflux,
    # factor=factor, sze3=sze3, jtot=jtot, jtot3=jtot3,
    # met_zl=met_zl, delz=delz, izref=izref,
    # ustar_ref=ustar_ref, ustar=ustar,
    # source=source, cncc=cncc, dispersion_np=dispersion_np
)

print(cncc)
