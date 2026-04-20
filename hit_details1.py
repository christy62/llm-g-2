import numpy as np
import pandas as pd
import time

# Load decay events

decays = pd.read_csv("/home/christy/helix_generator/decay_events/positron_decay_events_33k.csv")


# Helix propagator

class HelixFromCSV:

    def __init__(self, B=3.0):
        self.B = B

        # detector geometry (mm)
        self.RMIN = 70.0
        self.RMAX = 290.0
        self.ZMIN = -200.0
        self.ZMAX = 200.0

        self.m = 0.511  # MeV
        self.min_p = 10  # MeV

        # Vanes
        self.NVANE = 40
        self.VANE_THICKNESS = 0.32  # mm
        self.VANE_RADIUS = self.RMIN
        self.VANE_PHI_WIDTH = self.VANE_THICKNESS / self.VANE_RADIUS
        self.VANE_PHIS = np.linspace(0, 2*np.pi, self.NVANE, endpoint=False)

        # Material
        self.X0_mm = 150.0
        self.EC = 40.0  # MeV

    
    # Geometry np.random
    
    def in_vane(self, x, y):
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2*np.pi

        vane_index = int(phi / (2*np.pi) * self.NVANE)
        vane_phi = vane_index * (2*np.pi / self.NVANE)

        dphi = abs((phi - vane_phi + np.pi) % (2*np.pi) - np.pi)

        return dphi < self.VANE_PHI_WIDTH / 2

    
    
    # Energy loss 
    
    def bethe_bloch(self, E, step):
        
        beta = np.sqrt(1 - (self.m / E)**2)
        K = 3e-5
        return K * step * (1 / beta**2) * (np.log(2 * E / self.m) - beta**2)

    def bremsstrahlung(self, E, step):
         return 0.05 * E * (step / self.X0_mm)

   
    # Deterministic MCS
    
    def apply_mcs(self, px, py, pz, p, step):

        beta = p / np.sqrt(p**2 + self.m**2)
        theta0 = (13.6 / (beta * p)) * np.sqrt(step / self.X0_mm)

        # Deterministic scattering angle
        dtheta = theta0

        # Momentum direction
        ux, uy, uz = px/p, py/p, pz/p

        # Deterministic transverse direction
        # Construct a perpendicular vector using z-axis cross product
        vx = -uy
        vy = ux
        vz = 0.0

        norm = np.sqrt(vx*vx + vy*vy + vz*vz)
        if norm < 1e-12:
            return px, py, pz

        vx /= norm
        vy /= norm
        vz /= norm

        # Rotate momentum by fixed angle
        px_new = p * (ux*np.cos(dtheta) + vx*np.sin(dtheta))
        py_new = p * (uy*np.cos(dtheta) + vy*np.sin(dtheta))
        pz_new = p * uz

        return px_new, py_new, pz_new

    
    # Helper
    
    def compute_vane_energy_from_steps(self, step_vane, step_energy, n_vanes):
        vane_energy = np.zeros(n_vanes)
        for v, e in zip(step_vane, step_energy):
            vane_energy[v] += e
        return vane_energy

   
    # Recompute helix center after momentum changes
   
    def update_helix_center(self, x, y, px, py, p):
        pT = np.hypot(px, py)
        if pT < 1e-9:
            return None, None

        k = 0.2998
        R = (pT * 1e-3) / (k * self.B) * 1000
        xc = x + py / pT * R
        yc = y - px / pT * R
        return xc, yc

   
    # Helix propagation
   
    def compute_helix(self, decay_pos, decay_momentum,
                      step=0.3, max_steps=5000):

        x, y, z = decay_pos
        px, py, pz = decay_momentum
        p = np.sqrt(px**2 + py**2 + pz**2)
        c_mm_per_ns = 3e2   # speed of light in mm/ns

        # a detector threshold
        if p < self.min_p:
            return np.empty((0, 3)), 0.0, np.zeros(self.NVANE), np.array([]), np.array([]), np.array([])

        q = +1
        k = 0.2998  # GeV/(T·m)
        effective_step = min(step, self.VANE_THICKNESS)
        pts = []
        helix_vanes = []
        helix_cum_energy = []
        helix_time = []
        step_vane = []
        step_energy = []
        cumulative_energy = 0.0
        cumulative_time = 0.0
        c_mm = 3e11  # speed of light in mm/s
        pT = np.hypot(px, py)
        if pT < 1e-9:
            return np.empty((0, 3)), 0.0, np.zeros(self.NVANE), np.array([]), np.array([]), np.array([])

        pT_GeV = pT * 1e-3
        R = (pT_GeV / (k * self.B)) * 1000  # mm

        # Entry-only logic and accumulate material effects
        was_in_vane = False
        vane_path = 0.0
        vane_phi_entry = None
        vane_id_entry = None
        #recorded_vanes = set()  # Track which vanes have been recorded

        for _ in range(max_steps):

            # Update p and E from current momentum (ensures consistency)
            p = np.sqrt(px**2 + py**2 + pz**2)
            E = np.sqrt(p**2 + self.m**2)

            # Entry detection and path accumulation
            inside_vane = self.in_vane(x, y)

            # On vane ENTRY
            if inside_vane and not was_in_vane:
                vane_path = 0.0
                vane_phi_entry = np.arctan2(y, x) % (2*np.pi)
                vane_id_entry = int(vane_phi_entry / (2*np.pi) * self.NVANE) % self.NVANE

            # Magnetic bending
            pT = np.hypot(px, py)

            ds = effective_step if inside_vane else step
            beta = p / E
            dt_ns = ds / (beta * c_mm_per_ns)
            cumulative_time += dt_ns

            # While INSIDE vane, accumulate path (after ds is defined)
            if inside_vane:
                vane_path += ds

            # --- straight-line case ---
            if pT < 1e-9:
                x += (px / p) * ds
                y += (py / p) * ds
                z += (pz / p) * ds
                
                # Update state
                was_in_vane = inside_vane
                continue

            # convert pT to GeV
            pT_GeV = pT * 1e-3

            # helix radius
            R_m = pT_GeV / (k * self.B)
            R = R_m * 1000  # mm

            # Update helix center before rotation
            xc, yc = self.update_helix_center(x, y, px, py, p)
            if xc is None or yc is None:
                x += (px / p) * ds
                y += (py / p) * ds
                z += (pz / p) * ds
                
                # Update state
                was_in_vane = inside_vane
                continue
            
            # curvature (clockwise for positron, B = +z)
            dphi = - ds / R
            dphi = np.clip(dphi, -0.2, 0.2)

            # rotate position
            dx, dy = x - xc, y - yc
            cosd, sind = np.cos(dphi), np.sin(dphi)

            x = xc + dx * cosd - dy * sind
            y = yc + dx * sind + dy * cosd

            # rotate momentum
            px, py = (
                px * cosd - py * sind,
                px * sind + py * cosd
            )

            # re-normalize momentum
            norm = np.sqrt(px**2 + py**2 + pz**2)
            px *= p / norm
            py *= p / norm
            pz *= p / norm

            # z motion
            z += (pz / p) * ds

            # Check detector containment
            r = np.hypot(x, y)
            inside = (
                self.RMIN <= r <= self.RMAX and
                self.ZMIN <= z <= self.ZMAX
            )

            # Record hit on vane entry
            if inside and inside_vane and not was_in_vane:
                pts.append([x, y, z])
                helix_vanes.append(vane_id_entry)
                helix_cum_energy.append(cumulative_energy)
                helix_time.append(cumulative_time)


            # On vane EXIT: apply all material effects once
            if not inside_vane and was_in_vane:
                # Apply energy loss ONCE using current momentum
                p = np.sqrt(px**2 + py**2 + pz**2)
                E = np.sqrt(p**2 + self.m**2)

                #Energy loss
                if E < self.EC:
                    dE = self.bethe_bloch(E, vane_path)
                else:
                    dE = self.bremsstrahlung(E, vane_path)

                E -= dE

                if E <= self.m:
                    break
                p = np.sqrt(E**2 - self.m**2)
                cumulative_energy += dE

                step_vane.append(vane_id_entry)
                step_energy.append(dE)

                # Apply MCS ONCE
                
                px, py, pz = self.apply_mcs(px, py, pz, p, vane_path)

                # Renormalize momentum
                norm = np.sqrt(px**2 + py**2 + pz**2)
                px *= p / norm
                py *= p / norm
                pz *= p / norm

                # REQUIRED: Update helix center after momentum changes
                xc, yc = self.update_helix_center(x, y, px, py, p)

            # Update state at end of loop
            was_in_vane = inside_vane

            if abs(z) > self.ZMAX + 200:
                break

        vane_energy = self.compute_vane_energy_from_steps(
            step_vane, step_energy, self.NVANE
        )

        return (
            np.array(pts),
            0.0,
            vane_energy,
            np.array(helix_vanes),
            np.array(helix_cum_energy),
            np.array(helix_time)
        )


if __name__ == "__main__":
    # Run over events
    start_time = time.time()

    helix = HelixFromCSV(B=3.0)
    all_hits = []

    for i, (_, row) in enumerate(decays.iterrows()):

        event_id = int(row["event"])

        decay_pos = (row["x_decay"], row["y_decay"], row["z_decay"])
        decay_momentum = (row["px_decay"], row["py_decay"], row["pz_decay"])

        pts, _, _, vanes, cumE, times = helix.compute_helix(
            decay_pos,
            decay_momentum
        )

        for j in range(len(pts)):
            x, y, z = pts[j]
            all_hits.append([
                event_id,
                j,
                x, y, z,
                vanes[j],
                cumE[j],
                times[j]
            ])
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {i+1} events | Elapsed: {elapsed:.1f} s")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal runtime: {total_time:.2f} seconds")
    print(f"Average time per event: {total_time / len(decays):.5f} seconds")

    # Save hits

    hits_df = pd.DataFrame(
        all_hits,
        columns=["event", "hit_id", "x", "y", "z", "vane_id", "cum_energy_MeV", "time_ns"]
    )

    hits_df.to_csv("/home/christy/helix_generator/data_hits/detector_hits_33k.csv", index=False)
    print("Saved detector_hits_33k.csv")