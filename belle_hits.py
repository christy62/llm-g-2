import numpy as np
import pandas as pd


decays = pd.read_csv(
    "/home/christy/helix_generator/decay_events/positron_decay_events_100k.csv"
)


class HelixParams:

    def __init__(self, B=3.0):

        self.B = B
        self.m = 0.511

        # detector geometry
        self.RMIN = 90
        self.RMAX = 290
        self.ZMIN = -200
        self.ZMAX = 200


    def helix_parameters(self, x, y, z, px, py, pz, q=1):

        pT = np.hypot(px, py)

        if pT < 1e-9:
            return None

        pT_GeV = pT * 1e-3
        k = 0.2998

        R = (pT_GeV / (k * self.B)) * 1000

        xc = x + py/pT * R
        yc = y - px/pT * R

        D = np.hypot(xc, yc)

        if D < 1e-12:
            return None

        xp = xc - R * xc / D
        yp = yc - R * yc / D

        Lz = x*py - y*px
        sign = np.sign(Lz)

        d0 = sign * (D - R)

        phi_decay = np.arctan2(y - yc, x - xc)
        phi_poca = np.arctan2(yp - yc, xp - xc)

        dphi = (phi_poca - phi_decay + np.pi) % (2*np.pi) - np.pi
        s = R * dphi

        z0 = z + s * (pz/pT)

        phi0 = phi_poca + q*np.pi/2
        phi0 = (phi0 + np.pi) % (2*np.pi) - np.pi

        omega = q / R
        tan_lambda = pz / pT

        return d0, z0, phi0, omega, tan_lambda, R, xc, yc


    def reaches_detector(self, x, y, z, px, py, pz):

        step = 2.0
        max_steps = 3000

        for _ in range(max_steps):

            p = np.sqrt(px**2 + py**2 + pz**2)
            pT = np.hypot(px, py)

            if pT < 1e-9:
                return False

            k = 0.2998
            R = (pT*1e-3)/(k*self.B)*1000

            xc = x + py/pT * R
            yc = y - px/pT * R

            dphi = -step/R

            dx = x-xc
            dy = y-yc

            cosd = np.cos(dphi)
            sind = np.sin(dphi)

            x = xc + dx*cosd - dy*sind
            y = yc + dx*sind + dy*cosd

            z += (pz/p)*step

            r = np.hypot(x,y)

            if self.RMIN <= r <= self.RMAX and self.ZMIN <= z <= self.ZMAX:
                return True

            if abs(z) > self.ZMAX + 300:
                return False

        return False


if __name__ == "__main__":

    helix = HelixParams(B=3.0)

    rows = []

    for _, row in decays.iterrows():

        event = int(row["event"])

        x = row["x_decay"]
        y = row["y_decay"]
        z = row["z_decay"]

        px = row["px_decay"]
        py = row["py_decay"]
        pz = row["pz_decay"]

        # check if track reaches detector
        if not helix.reaches_detector(x, y, z, px, py, pz):
            continue

        params = helix.helix_parameters(x, y, z, px, py, pz)

        if params is None:
            continue

        d0, z0, phi0, omega, tanl, R, xc, yc = params

        rows.append([
            event,
            d0,
            z0,
            phi0,
            omega,
            tanl
        ])


    df = pd.DataFrame(
        rows,
        columns=[
            "event",
            "d0",
            "z0",
            "phi0",
            "omega",
            "tan_lambda"
        ]
    )

    df = df.round({
        "d0": 1,
        "z0": 1,
        "phi0": 2,
        "omega": 4,
        "tan_lambda": 2
    })

    df.to_csv(
        "/home/christy/helix_generator/data_hits/belle_params_detector_accepted_100k.csv",
        index=False
    )

    print("Saved belle_params_detector_accepted_100k.csv")