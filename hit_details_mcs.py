import numpy as np
import pandas as pd
import time


INPUT_CSV  = "/home/christy/helix_generator/decay_events/positron_decay_events_50k.csv"
OUTPUT_CSV = "/home/christy/helix_generator/data_hits/detector_hits_mcs_50k.csv"


class HelixFromCSV:

    def __init__(self, B=3.0):
        self.B = B

        # Detector geometry (mm)
        self.RMIN = 70.0
        self.RMAX = 290.0
        self.ZMIN = -200.0
        self.ZMAX = 200.0

        self.m     = 0.511   # MeV  (positron mass)
        self.min_p = 10.0    # MeV  (detector threshold)

        # Vanes
        self.NVANE          = 40
        self.VANE_THICKNESS = 0.32          # mm
        self.VANE_RADIUS    = self.RMIN
        self.VANE_PHI_WIDTH = self.VANE_THICKNESS / self.VANE_RADIUS
        self.VANE_PHIS      = np.linspace(0, 2 * np.pi, self.NVANE, endpoint=False)

        # Material properties
        self.X0_mm = 150.0
        self.EC    = 40.0   # MeV

    # Geometry

    def in_vane(self, x, y):
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi
        # FIX: % NVANE guards against phi == 2π floating-point edge case
        vane_index = int(phi / (2 * np.pi) * self.NVANE) % self.NVANE
        vane_phi   = vane_index * (2 * np.pi / self.NVANE)
        dphi = abs((phi - vane_phi + np.pi) % (2 * np.pi) - np.pi)
        return dphi < self.VANE_PHI_WIDTH / 2

    #  Energy loss 
    def bethe_bloch(self, E, step):
        beta = np.sqrt(max(1.0 - (self.m / E) ** 2, 1e-12))
        K    = 3e-5
        return K * step * (1.0 / beta ** 2) * (np.log(2.0 * E / self.m) - beta ** 2)

    def bremsstrahlung(self, E, step):
        return 0.05 * E * (step / self.X0_mm)

    #MCS with physical random scattering direction
    def apply_mcs(self, px, py, pz, p, step):

        beta = p / np.sqrt(p**2 + self.m**2)
        t    = step / self.X0_mm

        if t <= 0:
            return px, py, pz

        # --- Highland formula (improved theta0) ---
        theta0 = (13.6 / (beta * p)) * np.sqrt(t) * (1 + 0.038 * np.log(t))

        # --- Molière mixture ---
        # Fraction of tail events increases with thickness
        f_tail = min(0.15, 0.02 * np.log(1 + t * 100))  
        # (~2–10% typically, capped)

        if np.random.rand() < (1 - f_tail):
            # --- Gaussian core ---
            theta = np.random.normal(0.0, theta0)
        else:
            # --- Rutherford-like tail ---
            theta_min = theta0 * 0.5   # cutoff (screening)
            u = np.random.rand()
            theta = theta_min / np.sqrt(u)   # ~1/θ² tail
            theta *= np.random.choice([-1, 1])  # symmetric

        # --- Azimuth ---
        phi = np.random.uniform(0, 2*np.pi)

        # --- Direction basis ---
        ux, uy, uz = px / p, py / p, pz / p

        ref = np.array([1.0, 0.0, 0.0]) if abs(ux) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp1 = np.cross([ux, uy, uz], ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross([ux, uy, uz], perp1)

        vx = np.cos(phi) * perp1[0] + np.sin(phi) * perp2[0]
        vy = np.cos(phi) * perp1[1] + np.sin(phi) * perp2[1]
        vz = np.cos(phi) * perp1[2] + np.sin(phi) * perp2[2]

        cos_t, sin_t = np.cos(theta), np.sin(theta)

        new_px = p * (ux * cos_t + vx * sin_t)
        new_py = p * (uy * cos_t + vy * sin_t)
        new_pz = p * (uz * cos_t + vz * sin_t)

        return new_px, new_py, new_pz
    #  Helpers 

    def compute_vane_energy_from_steps(self, step_vane, step_energy, n_vanes):
        vane_energy = np.zeros(n_vanes)
        for v, e in zip(step_vane, step_energy):
            vane_energy[v] += e
        return vane_energy

    def update_helix_center(self, x, y, px, py, p):
        pT = np.hypot(px, py)
        if pT < 1e-9:
            return None, None
        k  = 0.2998
        R  = (pT * 1e-3) / (k * self.B) * 1000  # mm
        xc = x + py / pT * R
        yc = y - px / pT * R
        return xc, yc

    #  Main propagation

    def compute_helix(self, decay_pos, decay_momentum,  
                      step=0.3, max_steps=5000):

        x, y, z     = decay_pos
        px, py, pz  = decay_momentum
        p           = np.sqrt(px**2 + py**2 + pz**2)
        c_mm_per_ns = 3e2   # speed of light in mm/ns

        if p < self.min_p:
            return (np.empty((0, 3)), 0.0, np.zeros(self.NVANE),
                    np.array([]), np.array([]), np.array([]))

        k = 0.2998   # GeV/(T·m)

        pts              = []
        helix_vanes      = []
        helix_cum_energy = []
        helix_time       = []
        step_vane        = []
        step_energy      = []
        cumulative_energy = 0.0
        cumulative_time   = 0.0

        pT = np.hypot(px, py)
        if pT < 1e-9:
            return (np.empty((0, 3)), 0.0, np.zeros(self.NVANE),
                    np.array([]), np.array([]), np.array([]))

        was_in_vane    = False
        vane_path      = 0.0
        vane_phi_entry = None
        vane_id_entry  = None

        for _ in range(max_steps):

            p = np.sqrt(px**2 + py**2 + pz**2)
            E = np.sqrt(p**2 + self.m**2)

            inside_vane = self.in_vane(x, y)

            #  Vane ENTRY bookkeeping 
            if inside_vane and not was_in_vane:
                vane_path      = 0.0
                vane_phi_entry = np.arctan2(y, x) % (2 * np.pi)
                vane_id_entry  = int(vane_phi_entry / (2 * np.pi) * self.NVANE) % self.NVANE

            pT   = np.hypot(px, py)
            beta = p / E

            # FIX: dynamic step to cap rotation angle for low-p tracks
            if pT > 1e-9:
                pT_GeV = pT * 1e-3
                R      = (pT_GeV / (k * self.B)) * 1000   # mm
                ds     = min(step if not inside_vane else min(step, self.VANE_THICKNESS),
                             R * 0.05)
            else:
                ds = step if not inside_vane else min(step, self.VANE_THICKNESS)

            dt_ns          = ds / (beta * c_mm_per_ns)
            cumulative_time += dt_ns

            if inside_vane:
                vane_path += ds

            #  Propagate position & momentum 
            if pT < 1e-9:
                # Straight-line fallback
                x += (px / p) * ds
                y += (py / p) * ds
                z += (pz / p) * ds
                was_in_vane = inside_vane
                continue

            pT_GeV = pT * 1e-3
            R      = (pT_GeV / (k * self.B)) * 1000   # mm

            xc, yc = self.update_helix_center(x, y, px, py, p)
            if xc is None:
                x += (px / p) * ds
                y += (py / p) * ds
                z += (pz / p) * ds
                was_in_vane = inside_vane
                continue

            dphi       = -ds / R                       # curvature sign for positron
            dx, dy     = x - xc, y - yc
            cosd, sind = np.cos(dphi), np.sin(dphi)

            x  = xc + dx * cosd - dy * sind
            y  = yc + dx * sind + dy * cosd
            px_new = px * cosd - py * sind
            py_new = px * sind + py * cosd
            px, py = px_new, py_new

            # Re-normalise transverse momentum magnitude
            norm = np.sqrt(px**2 + py**2 + pz**2)
            px  *= p / norm
            py  *= p / norm
            pz  *= p / norm

            z += (pz / p) * ds

            # Check containment 
            r      = np.hypot(x, y)
            inside = (self.RMIN <= r <= self.RMAX and self.ZMIN <= z <= self.ZMAX)

            # FIX: record hit AFTER position update, on vane ENTRY
            if inside and inside_vane and not was_in_vane:
                pts.append([x, y, z])
                helix_vanes.append(vane_id_entry)
                helix_cum_energy.append(cumulative_energy)
                helix_time.append(cumulative_time)

            # Vane EXIT: apply material effects once 
            if not inside_vane and was_in_vane:
                p = np.sqrt(px**2 + py**2 + pz**2)
                E = np.sqrt(p**2 + self.m**2)

                dE = self.bethe_bloch(E, vane_path) if E < self.EC else \
                     self.bremsstrahlung(E, vane_path)
                E -= dE

                if E <= self.m:
                    break

                p                  = np.sqrt(E**2 - self.m**2)
                cumulative_energy += dE
                step_vane.append(vane_id_entry)
                step_energy.append(dE)

                px, py, pz = self.apply_mcs(px, py, pz, p, vane_path)

                norm = np.sqrt(px**2 + py**2 + pz**2)
                px  *= p / norm
                py  *= p / norm
                pz  *= p / norm

                xc, yc = self.update_helix_center(x, y, px, py, p)

            was_in_vane = inside_vane

            if abs(z) > self.ZMAX + 200:
                break

        vane_energy = self.compute_vane_energy_from_steps(
            step_vane, step_energy, self.NVANE)

        return (
            np.array(pts),
            0.0,
            vane_energy,
            np.array(helix_vanes),
            np.array(helix_cum_energy),
            np.array(helix_time),
        )


#  Entry point 
if __name__ == "__main__":
    decays     = pd.read_csv(INPUT_CSV)
    start_time = time.time()
    helix    = HelixFromCSV(B=3.0)
    all_hits = []

    for i, (_, row) in enumerate(decays.iterrows()):
        event_id       = int(row["event"])
        decay_pos      = (row["x_decay"], row["y_decay"], row["z_decay"])
        decay_momentum = (row["px_decay"], row["py_decay"], row["pz_decay"])

        pts, _, _, vanes, cumE, times = helix.compute_helix(
        decay_pos, decay_momentum
    )

        for j in range(len(pts)):
            x, y, z = pts[j]
            all_hits.append([event_id, j, x, y, z,
                             vanes[j], cumE[j], times[j]])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i+1} events | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"\nTotal runtime   : {total_time:.2f}s")
    print(f"Avg per event   : {total_time / len(decays):.5f}s")

    hits_df = pd.DataFrame(
        all_hits,
        columns=["event", "hit_id", "x", "y", "z",
                 "vane_id", "cum_energy_MeV", "time_ns"]
    )
    hits_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")