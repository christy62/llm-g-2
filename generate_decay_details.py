import numpy as np
import pandas as pd

#rng = np.random.default_rng(42)
rng = np.random.default_rng()

# ============================================================
# CONSTANTS
# ============================================================

# Detector (only used for acceptance later if needed)
RMIN, RMAX = 70.0, 290.0   # mm
ZMIN, ZMAX = -200.0, 200.0

# Magnetic field
B = 3.0                   # Tesla
K = 0.2998                # MeV / (mm·T)

# Particle masses
M_E  = 0.511              # MeV (positron)
M_MU = 105.658            # MeV (muon)

Q = +1                    # positron charge

# ------------------------------------------------------------
# Muon injection parameters 
# ------------------------------------------------------------

MUON_X0 = -333.564        # mm
MUON_Y0 = 0.0             # mm
MUON_Z0 = 0.0             # mm

# Muon momentum direction at injection
MUON_PX0 = 0.0
MUON_PY0 = 1.0
MUON_PZ0 = 0.0

# Muon momentum magnitude 
P_MU_MEAN = 300.0         # MeV/c
SIGMA_P_MU = 0.5          # small spread

# Vertical beam size
SIGMA_Z_MU = 50.0         # mm


# ============================================================
# MUON ORBIT GEOMETRY 
# ============================================================

def compute_orbit_center(x, y, px, py):
    pT = np.hypot(px, py)
    R = pT / (K * B)

    tx, ty = px / pT, py / pT

    # p̂ × B̂ for B = +z
    nx, ny = ty, -tx

    # Note THE MINUS SIGN
    xc = x + Q * R * nx
    yc = y - Q * R * ny

    return xc, yc, R





# MUON ORBIT (CENTERED AT ORIGIN)


R_MU = abs(MUON_X0)   # since MUON_X0 = -333.564 mm

def generate_positron_decay_vertex():

    phi0 = np.arctan2(MUON_Y0, MUON_X0)

    tau_mu = 6.6e-6      # s
    c_mm = 3e11          # mm/s

    t_decay = rng.exponential(tau_mu)
    dphi = (c_mm * t_decay) / R_MU

    phi = (phi0 + dphi) % (2*np.pi)

    x = R_MU * np.cos(phi)
    y = R_MU * np.sin(phi)
    z = 0.0

    assert np.isclose(x*x + y*y, R_MU*R_MU, rtol=1e-6)

    return x, y, z



# POSITRON DECAY MOMENTUM 


def generate_positron_momentum():
    """
    Unpolarized Michel spectrum
    """
    Emax = 300  # MeV
    while True:
        x = rng.random()
        y = rng.random()
        if y < x*x*(3 - 2*x):
            E = x * Emax
            return np.sqrt(E*E - M_E*M_E)


def generate_positron_direction():
    cost = rng.uniform(-1, 1)
    sint = np.sqrt(1 - cost*cost)
    phi = rng.uniform(0, 2*np.pi)
    return sint*np.cos(phi), sint*np.sin(phi), cost



# EVENT GENERATION


def generate_event(event_id):
    """
    Generate ONE positron decay.
    Returns decay vertex + decay momentum (positron).
    """

    # Decay vertex (from muon orbit)
    x, y, z = generate_positron_decay_vertex()

    # Positron momentum
    p = generate_positron_momentum()
    ux, uy, uz = generate_positron_direction()

    px = p * ux
    py = p * uy
    pz = p * uz

    return [
        event_id,
        x, y, z,
        px, py, pz,
        p
    ]



# MAIN LOOP


def run_simulation(N_EVENTS=50000):

    events = []

    for i in range(N_EVENTS):
        events.append(generate_event(i))

        if (i + 1) % 10000 == 0:
            print(f"Generated {i+1} events")

    df = pd.DataFrame(
        events,
        columns=[
            "event",
            "x_decay", "y_decay", "z_decay",
            "px_decay", "py_decay", "pz_decay",
            "p_decay"
        ]
    )

    df.to_csv("/home/christy/helix_generator/decay_events/positron_decay_events_random_50k.csv", index=False)
    print("Saved positron_decay_events_random_50k.csv")



# RUN


if __name__ == "__main__":
    run_simulation(50000)