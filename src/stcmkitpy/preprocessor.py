#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STCM data pre-processor
Refactored into a class structure for eventual PyPI packaging.

Updates:
1. save_prep_data() writes 7 columns (including the column of 1s).
2. Added transform_orientation_quaternion() method.
"""

import numpy as np
import pandas as pd
import os
import re
import ppigrf  # from https://pypi.org/project/ppigrf/
from datetime import datetime


class STCMPreprocessor:
    """
    Class to read, parse, and pre-process .stcm data,
    compute (optionally scaled) IGRF components,
    apply heading/pitch/roll transformations,
    and compute calibration coefficients via a least-squares approach.

    Attributes
    ----------
    fn : str
        Path to input .stcm file.
    df : pd.DataFrame
        Main data after loading.
    dg : np.ndarray
        Array of shape (N,3) for raw magnetometer or sensor data.
    rpyf : np.ndarray
        Array of shape (N,3) for the "transformed field" data (sine/cosine approach).
    rpyf_quat : np.ndarray
        Array of shape (N,3) for the "transformed field" data (quaternion approach).
    coeff : np.ndarray
        Fitted coefficient array of shape (3,3).
    intercept : np.ndarray
        Intercept terms for each output dimension (shape (3,)).
    score : float
        Average R^2 score across the 3 outputs (sine/cosine fit).
    btratio : float
        Ratio between proton total field (if any) and IGRF total field.
    """

    def __init__(self, fn, bt_proton=1.0):
        """
        Parameters
        ----------
        fn : str
            Full path to .stcm file.
        bt_proton : float, optional
            Proton-derived total magnetic field. 1.0 means "no proton data" scenario.
        """
        self.fn = fn
        self.bt_proton = bt_proton

        self.df = None
        self.dg = None
        self.rpyf = None
        self.rpyf_quat = None

        self.coeff = None
        self.intercept = None
        self.score = None
        self.btratio = None

        # IGRF base components (east,north,down)
        self.Be_igrf = None
        self.Bn_igrf = None
        self.Bd_igrf = None
        self.Bt_igrf = None

    def load_data(self):
        """
        Read the .stcm file, handle spacing, parse into a DataFrame,
        and create a 'datetime' column.
        """
        print("STEP1: Reading STCM file =>", self.fn)
        fname_base = os.path.basename(self.fn)
        file_path = os.path.dirname(self.fn)
        file_name = os.path.splitext(fname_base)[0]

        # Fix irregular spacing
        with open(self.fn, 'r') as f:
            text = f.read()
        tmptext = re.sub('  ', ' ', text)

        # Write to temp file
        out1 = os.path.join(file_path, file_name + '_temp.dat')
        with open(out1, 'w') as f:
            f.write(tmptext)

        # Read into DataFrame
        names = [
            'Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
            'Lat', 'Lon', 'Depth', 'Proton', 'Gravity',
            'Hx', 'Hy', 'Hz', 'Heading', 'Roll', 'Pitch'
        ]
        df = pd.read_csv(out1, header=None, sep=' ', engine='python')
        df.columns = names

        # Convert "Heading" to numeric and fill possible NaN
        df['Heading'] = pd.to_numeric(df['Heading'], errors='coerce')
        if df['Heading'].isnull().any():
            print("Warning: There are NaN in 'Heading' -> filling with 0.")
            df['Heading'].fillna(0, inplace=True)

        self.df = df

        # Prepare a datetime if you need it:
        tr = pd.to_datetime(
            {
                'year': df['Year'],
                'month': df['Month'],
                'day': df['Day'],
                'hour': df['Hour'],
                'minute': df['Min'],
                'second': df['Sec'],
            }
        )
        self.df['datetime'] = tr
        print("...Data loaded, shape:", self.df.shape)

    def compute_igrf(self, row_index=0):
        """
        Compute or fetch scaled IGRF vectors based on the first row's lat/lon/time,
        then set btratio if proton data is present.
        """
        print("STEP2: IGRF computation")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Use the first row's time/lat/lon for demonstration.
        lat = float(self.df.loc[row_index, 'Lat'])
        lon = float(self.df.loc[row_index, 'Lon'])
        # Typically you'd want depth in km above sea level: here we set h=0
        h = 0.0

        dt = self.df.loc[row_index, 'datetime']

        # ppigrf returns (Be, Bn, Bu) in nT => (east, north, up)
        Be, Bn, Bu = ppigrf.igrf(lon, lat, h, dt)
        Be = float(Be)
        Bn = float(Bn)
        Bu = float(Bu)
        Bd = -Bu  # convert "up" to "down"

        # Magnitude
        Bt = np.sqrt(Be**2 + Bn**2 + Bd**2)

        # If we have real proton data
        if abs(self.bt_proton - 1.0) < 1e-6:
            # No real proton data
            print("...No proton data => ratio=1.0")
            self.btratio = 1.0
        else:
            # Scale IGRF to match proton data
            print("...Have proton data => adjusting IGRF")
            self.btratio = self.bt_proton / Bt
            Be *= self.btratio
            Bn *= self.btratio
            Bd *= self.btratio

        # Store them as attributes if you need them later
        self.Be_igrf = Be
        self.Bn_igrf = Bn
        self.Bd_igrf = Bd
        self.Bt_igrf = Bt
        
        if np.isscalar(Bt):
            print(f"IGRF total field = {Bt:.4f} nT;  Adjusted ratio = {self.btratio:.4f}")
        else:
            print(f"IGRF total field (mean) = {np.mean(Bt):.4f} nT;  Adjusted ratio = {self.btratio:.4f}")

    # =========================================================================
    # CONVENTIONAL TRANSFORM
    # =========================================================================


    def transform_orientation(self):
        """
        Apply heading/pitch/roll to the base IGRF vector for *each row*
        using the explicit sine/cosine expansions from the original code.
        """
        print("STEP3: Orientation transform (roll/pitch/heading)")

        if self.df is None:
            raise ValueError("Data not loaded.")
        if self.Be_igrf is None:
            raise ValueError("IGRF not computed. Call compute_igrf() first.")

        # For simplicity, build rpyf for each row using the same Be/Bn/Bd
        cos_Ro = np.cos(np.deg2rad(self.df['Roll'].values))
        cos_Pi = np.cos(np.deg2rad(self.df['Pitch'].values))
        cos_He = np.cos(np.deg2rad(self.df['Heading'].values))
        sin_Ro = np.sin(np.deg2rad(self.df['Roll'].values))
        sin_Pi = np.sin(np.deg2rad(self.df['Pitch'].values))
        sin_He = np.sin(np.deg2rad(self.df['Heading'].values))

        rpyf = np.zeros((len(self.df), 3))

        # Use the same IGRF for all rows (some people do row-by-row for lat/lon/time)
        Bn, Be, Bd = self.Bn_igrf, self.Be_igrf, self.Bd_igrf

        rpyf[:, 0] = (cos_He * cos_Pi) * Bn + (cos_Pi * sin_He) * Be + (sin_Pi * -1) * Bd
        rpyf[:, 1] = (
            (-cos_Ro * sin_He + cos_He * sin_Pi * sin_Ro) * Bn
            + (cos_He * cos_Ro + sin_He * sin_Pi * sin_Ro) * Be
            + (cos_Pi * sin_Ro) * Bd
        )
        rpyf[:, 2] = (
            (cos_He * cos_Ro * sin_Pi + sin_He * sin_Ro) * Bn
            + (cos_Ro * sin_He * sin_Pi - cos_He * sin_Ro) * Be
            + (cos_Pi * cos_Ro) * Bd
        )

        self.rpyf = rpyf
        # Meanwhile, "dg" is presumably the raw sensor reading columns
        self.dg = self.df[['Hx', 'Hy', 'Hz']].values  # shape (N,3)
        print("...Orientation transformed. rpyf shape:", rpyf.shape)

    # =========================================================================
    # QUATERNION-BASED TRANSFORM
    # =========================================================================

    def transform_orientation_quaternion(self):
        """
        OPTIONAL: A quaternion-based orientation transform using heading, pitch, and roll
                  (treated as yaw, pitch, roll in that order).
        Stores the result in self.rpyf_quat (shape N,3).
        """

        if self.df is None:
            raise ValueError("Data not loaded.")
        if self.Be_igrf is None:
            raise ValueError("IGRF not computed. Call compute_igrf() first.")

        # We'll do a row-by-row transform
        n_data = len(self.df)
        rpyf_q = np.zeros((n_data, 3))

        for i in range(n_data):
            # heading ~ yaw, pitch, roll
            yaw = np.deg2rad(self.df.loc[i, 'Heading'])
            pitch = np.deg2rad(self.df.loc[i, 'Pitch'])
            roll = np.deg2rad(self.df.loc[i, 'Roll'])

            # Convert Euler -> quaternion
            q = self._euler_to_quaternion(roll, pitch, yaw)

            # The "base" IGRF vector to rotate
            # (If you prefer row-specific lat/lon/time, you'd compute row by row.)
            vec_igrf = np.array([self.Bn_igrf, self.Be_igrf, self.Bd_igrf])

            # Rotate the vector
            rotated_vec = self._rotate_vector_by_quaternion(vec_igrf, q)
            rpyf_q[i, :] = rotated_vec

        self.rpyf_quat = rpyf_q

        # Also define self.dg if it is not defined
        if self.dg is None:
            self.dg = self.df[['Hx', 'Hy', 'Hz']].values

        print("Quaternion-based orientation transform complete. rpyf_quat shape:", rpyf_q.shape)

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        # ← roll, pitch, yaw の順に変更！
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])


    @staticmethod
    def _q_conjugate(q):
        """Conjugate of quaternion q=[w,x,y,z]."""
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def _q_mult(q1, q2):
        """
        Multiply two quaternions q1*q2.
        q=[w,x,y,z].
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        return np.array([w, x, y, z])


    def _rotate_vector_by_quaternion(self, vec, q):
        vq = np.array([0.0, vec[0], vec[1], vec[2]])
        tmp = self._q_mult(self._q_conjugate(q), vq)
        rotated = self._q_mult(tmp, q)
        return rotated[1:]

    # =========================================================================
    # Fit to IGRF
    # =========================================================================

    def fit_model(self):
        """
        Perform a linear regression-like solution (via np.linalg.lstsq)
        on dg => rpyf (the sine/cosine-based transform).
        We include an intercept by adding a column of ones to dg.
        """
        print("STEP4: Fitting model via lstsq (like scikit-learn's LinearRegression)")

        if self.dg is None or self.rpyf is None:
            raise ValueError("Data or orientation transform is missing.")

        X = self.dg  # shape (N,3)
        Y = self.rpyf  # shape (N,3)

        # Add a column of ones to handle the intercept
        ones_col = np.ones((X.shape[0], 1))
        X_ = np.hstack([X, ones_col])  # shape (N,4)

        # Solve for Beta => shape (4,3)
        Beta, residuals, rank, s = np.linalg.lstsq(X_, Y, rcond=None)

        # The last row of Beta is the intercept
        self.coeff = Beta[:3, :].T  # shape (3,3) -> each row is the output dimension
        self.intercept = Beta[3, :]  # shape (3,)

        # Compute average R^2 across the 3 outputs
        Yhat = X_ @ Beta  # shape(N,3)
        r2_list = []
        for j in range(Y.shape[1]):
            y_true = Y[:, j]
            y_pred = Yhat[:, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot > 0:
                r2_list.append(1.0 - ss_res / ss_tot)
            else:
                r2_list.append(0.0)

        self.score = np.mean(r2_list)
        print("...Fit complete. R^2 =", self.score)

    def save_prep_data(self, outname=None):
        """
        Save the 'prep' data.  By request, we need 7 columns:
          - Hx, Hy, Hz, 1, rpyf(3)
        in that exact order, matching the original approach.
        """
        if outname is None:
            # default: same folder, same base name + '.prep'
            base = os.path.splitext(os.path.basename(self.fn))[0]
            outname = os.path.join(os.path.dirname(self.fn), base + '.prep')

        if self.dg is None or self.rpyf is None:
            raise ValueError("No data to save. Run transform steps first.")

        # 1) shape (N,3) => [Hx,Hy,Hz]
        # 2) add constant column (N,1)
        ones_col = np.ones((self.dg.shape[0], 1))

        # 3) append rpyf => shape (N,3)
        # final shape => (N, 7)
        data_7col = np.hstack((self.dg, ones_col, self.rpyf))

        # Format: 7 columns: Hx, Hy, Hz, 1, rpyf(x3)
        # Original code used ('%07.1f','%07.1f','%07.1f','%01d','%07.1f','%07.1f','%07.1f')
        np.savetxt(
            outname,
            data_7col,
            fmt=('%07.1f','%07.1f','%07.1f','%01d','%07.1f','%07.1f','%07.1f'),
            delimiter=' '
        )
        print(f"Prep data (7 columns) saved to {outname}")

    def print_results(self):
        """
        Print out the final fitting coefficients and relevant metrics
        in a style similar to the original code snippet.
        """
        print("--RESULT--")
        if self.coeff is not None:
            print("coeff:\n", self.coeff)
        if self.score is not None:
            print("R^2 score =", self.score)
        if self.intercept is not None:
            print("Intercept =", self.intercept)
        if self.btratio is not None:
            print("Bt ratio (IGRF vs Proton) =", self.btratio)
