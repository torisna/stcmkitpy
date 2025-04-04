#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STCMConverter

- Loads previously obtained coeff.npy and intercept.npy
- Applies them to raw stcm data to compute FLD in sensor coords
- Transforms to Earth coordinates
- Applies a lowpass Butterworth filter
- Resamples the data
- Computes distance traveled
- Computes IGRF for each timestamp, latitude, longitude
- Outputs .anm file with the "anomalies" (ANM) = FLD - IGRF

Note: In your original code, there is `igrf.igrf(...)` but also `import ppigrf`.
      Here we use ppigrf.igrf() consistently.
"""

import os
import glob
import time

import numpy as np
import pandas as pd

from scipy import signal
from geopy import distance
from datetime import datetime

import ppigrf  # from "pip install ppigrf"

class STCMConverter:
    """
    Convert raw STCM files to .anm files using existing calibration 
    coefficients (coeff.npy, intercept.npy) and a lowpass filter approach.

    Typical usage:
      converter = STCMConverter(
          coeff_path="GS24coeff.npy",
          intercept_path="GS24coeff_intercept.npy",
          filter_order=5, 
          filter_cutoff=0.008, 
          fs=10
      )
      converter.process_directory("/path/to/lines/")
    """

    def __init__(self,
                 coeff_path,
                 intercept_path,
                 filter_order=5,
                 filter_cutoff=0.008,
                 fs=10,
                 resample_rule='1s'):
        """
        Parameters
        ----------
        coeff_path : str
            Path to the .npy file containing calibration coefficients (3x3 matrix).
        intercept_path : str
            Path to the .npy file containing intercept (shape(3,)).
        filter_order : int
            Butterworth filter order.
        filter_cutoff : float
            Normalized cutoff frequency (0~1). Typically < 0.5 for lowpass.
            e.g., if fs=10 Hz, nyquist=5, cutoff=0.008 => actual freq ~ 5*0.008=0.04 Hz
        fs : float
            Sampling frequency in Hz (used for normalizing the filter).
        resample_rule : str
            Resample frequency for Pandas (e.g. '1s' for 1 second).
        """
        self.coeff_path = coeff_path
        self.intercept_path = intercept_path

        # Load calibration
        self.coeff = np.load(self.coeff_path)      # shape(3,3)
        self.intercept = np.load(self.intercept_path)  # shape(3,)

        # Filter params
        self.filter_order = filter_order
        self.filter_cutoff = filter_cutoff
        self.fs = fs
        self.nyquist = self.fs / 2.0
        self.resample_rule = resample_rule

    def process_directory(self, directory):
        """
        Process all .stcm files in the given directory, output .anm files.
        """
        # Gather stcm files
        stcm_files = [os.path.join(directory, f) 
                      for f in os.listdir(directory) 
                      if f.endswith('.stcm')]

        if not stcm_files:
            print("No .stcm files found in", directory)
            return

        print(f"--- Start processing {len(stcm_files)} stcm files in: {directory} ---")
        for fn in stcm_files:
            self.process_single_file(fn)
        print("--- All done. ---")

    def process_single_file(self, fn):
        """
        Process one .stcm file end-to-end:
          1. Load .stcm into DataFrame
          2. Convert to FLD sensor coords (using coeff / intercept)
          3. Transform to Earth coords
          4. Lowpass filter
          5. Resample to 1s
          6. Compute distance
          7. Compute IGRF
          8. Compute ANM
          9. Save .anm
        """
        start_time = time.time()

        base_name = os.path.splitext(os.path.basename(fn))[0]
        dir_path = os.path.dirname(fn)
        anm_file_name = os.path.join(dir_path, base_name + ".anm")

        print(f"\nProcessing file: {fn}")

        # 1) Load stcm
        try:
            df = self._load_stcm(fn)
        except pd.errors.EmptyDataError:
            print(f"  -> File {fn} is empty. Skipping.")
            return
        except Exception as e:
            print(f"  -> Error reading file {fn}: {e}")
            return

        if df.empty:
            print(f"  -> DataFrame is empty after load. Skipping.")
            return

        # 2) Convert to FLD sensor coords
        #    (apply calibration matrix "coeff" and intercept)
        prepfld = self._apply_coeff_intercept(df)

        # 回転行列変換
        fld_mat = self._sensor_to_earth(df, prepfld)



        # # 3) Transform to Earth coords
        # #    (heading, roll, pitch -> Earth-based FLD)
        # # fld_earth = self._sensor_to_earth(df, prepfld)
        fld_earth = self._sensor_to_earth_quaternion_vectorized(df, prepfld)

        # # 差分のRMSE（Root Mean Square Error）
        # diff = fld_earth[:, :3] - fld_mat[:, :3]
        # rmse = np.sqrt(np.mean(diff**2))
        # print(f"🌀 RMSE (quaternion vs matrix): {rmse:.3f} nT")
        # # 差分保存（任意）
        # diff_outfile = os.path.join(dir_path, base_name + "_quat_diff.csv")
        # pd.DataFrame(diff, columns=["dx", "dy", "dz"]).to_csv(diff_outfile, index=False)
        # print(f"📝 差分ファイル保存: {diff_outfile}")


        # 4) Lowpass filter
        fld_filtered = self._lowpass_filter(fld_earth)

        # 5) Resample data to 1s
        dj = self._make_dataframe_and_resample(df, fld_filtered)

        # 6) Compute distances
        dj = self._compute_distance(dj)

        # 7) Compute IGRF
        dj = self._compute_igrf(dj)

        # 8) Compute ANM
        dj = self._compute_anm(dj)

        # 9) Save .anm
        dj.to_csv(anm_file_name, index=True)
        print(f"  -> Saved: {anm_file_name}")

        elapsed = time.time() - start_time
        print(f"  -> Done in {elapsed:.2f} s.")

    # ----------------------------------------------------------------
    # Utility / stepwise methods
    # ----------------------------------------------------------------
    def _load_stcm(self, fn):
        """
        Read .stcm file into a DataFrame with standard columns:
        [Year, Month, Day, Hour, Min, Sec, Lat, Lon, Depth,
         Proton, Gravity, Hx, Hy, Hz, Heading, Roll, Pitch]
        """
        names = [
            'Year','Month','Day','Hour','Min','Sec',
            'Lat','Lon','Depth','Proton','Gravity',
            'Hx','Hy','Hz','Heading','Roll','Pitch'
        ]
        df = pd.read_csv(fn, header=None, sep=' ', engine='python')
        df.columns = names

        # Convert numeric if needed
        for col in names:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Make a datetime index or column
        dt = pd.to_datetime({
            'year': df['Year'],
            'month': df['Month'],
            'day': df['Day'],
            'hour': df['Hour'],
            'minute': df['Min'],
            'second': df['Sec']
        })
        df['datetime'] = dt

        return df

    def _apply_coeff_intercept(self, df):
        """
        Using columns [Hx, Hy, Hz], multiply by 'coeff' and then add 'intercept'.
        Returns array shape (N,4) = [fldX, fldY, fldZ, magnitude].
        """
        raw_xyz = df[['Hx','Hy','Hz']].values  # shape (N,3)

        # Multiply by coeff => shape (N,3)
        #   prepfld[:,0:3] = raw_xyz dot coeff + intercept
        #   but note: raw_xyz shape(N,3), coeff shape(3,3)
        # => (N,3)
        fld_sens = np.dot(raw_xyz, self.coeff.T)  # or raw_xyz @ coeff.T
        # Then add intercept => broadcast
        fld_sens += self.intercept

        # Magnitude
        mag = np.sqrt(np.sum(fld_sens**2, axis=1))

        # Combine
        prepfld = np.hstack([fld_sens, mag.reshape(-1,1)])  # shape(N,4)
        return prepfld

    def _sensor_to_earth(self, df, prepfld):
        """
        Transform sensor-coord FLD to Earth coords using heading/pitch/roll.
        Return shape (N,4): [X, Y, Z, magnitude] in Earth coords.
        """
        # rename for clarity
        hx_sens = prepfld[:,0]
        hy_sens = prepfld[:,1]
        hz_sens = prepfld[:,2]

        # Precompute sin/cos
        roll_rad  = np.deg2rad(df['Roll'].values)
        pitch_rad = np.deg2rad(df['Pitch'].values)
        hdg_rad   = np.deg2rad(df['Heading'].values)

        cos_rol = np.cos(roll_rad)
        cos_pth = np.cos(pitch_rad)
        cos_hdg = np.cos(hdg_rad)
        sin_rol = np.sin(roll_rad)
        sin_pth = np.sin(pitch_rad)
        sin_hdg = np.sin(hdg_rad)

        # Inverse rotation matrix components (similar to your code)
        invRPY11 = cos_hdg * cos_pth
        invRPY12 = sin_hdg * cos_rol + cos_hdg * sin_pth * sin_rol
        invRPY13 = sin_hdg * sin_rol - cos_hdg * sin_pth * cos_rol

        invRPY21 = -sin_hdg * cos_pth
        invRPY22 = cos_hdg*cos_rol - sin_hdg*sin_pth*sin_rol
        invRPY23 = cos_hdg*sin_rol + sin_hdg*sin_pth*cos_rol

        invRPY31 = sin_pth
        invRPY32 = -cos_pth * sin_rol
        invRPY33 = cos_pth * cos_rol

        # Earth coords
        ex = invRPY11*hx_sens + invRPY12*hy_sens + invRPY13*hz_sens
        ey = invRPY21*hx_sens + invRPY22*hy_sens + invRPY23*hz_sens
        ez = invRPY31*hx_sens + invRPY32*hy_sens + invRPY33*hz_sens

        mag_earth = np.sqrt(ex**2 + ey**2 + ez**2)

        fld_earth = np.column_stack([ex, ey, ez, mag_earth])  # shape(N,4)
        return fld_earth

    def _sensor_to_earth_quaternion_vectorized(self, df, prepfld):
        """
        Transform sensor-coord FLD (prepfld) to Earth coords using quaternions,
        in a fully vectorized manner (no Python-level for-loops).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns 'Heading', 'Roll', 'Pitch' (degrees).
            Length = N.
        prepfld : np.ndarray, shape (N,4)
            [fldX, fldY, fldZ, magnitude_sens]
            We only use the first 3 columns (sensor vectors).
        
        Returns
        -------
        fld_earth : np.ndarray, shape (N,4)
            [ex, ey, ez, magnitude_earth], Earth coords
        """
        # 1) Prepare Euler angles as arrays
        heading_deg = df['Heading'].values  # shape(N,)
        pitch_deg   = df['Pitch'].values    # shape(N,)
        roll_deg    = df['Roll'].values     # shape(N,)

        # 2) Convert Euler -> Quaternions (all rows at once)
        qs = self.euler_to_quaternion_array(heading_deg, pitch_deg, roll_deg)  # shape (N,4)

        # 3) Rotate sensor vectors by these quaternions
        #    sensor vector = prepfld[:, 0:3]
        sensor_vecs = prepfld[:, 0:3]  # shape (N,3)
        earth_vecs  = self.rotate_vectors_by_quaternions_array(sensor_vecs, qs)  # shape (N,3)

        # 4) Magnitude
        mag_earth = np.sqrt(np.sum(earth_vecs**2, axis=1))  # shape(N,)

        # 5) Combine into final array (N,4)
        fld_earth = np.column_stack([earth_vecs, mag_earth])
        return fld_earth


    # ------------------------------------------------------------------------
    # 下記はベクトル演算のためのヘルパー関数群。1回の呼び出しで N 個のクォータニオン計算をまとめて処理します。
    # ------------------------------------------------------------------------

    def euler_to_quaternion_array(self, heading_deg, pitch_deg, roll_deg):
        """
        heading, pitch, roll (deg) をまとめてクォータニオンに変換する (ベクトル演算版)。
        配列の形状はすべて (N,) で、出力は (N,4) = [w,x,y,z].
        
        Convention: 
        heading ~ yaw (Z), pitch (Y), roll (X)
        """
        # Convert deg -> rad
        # shape(N,)
        yaw   = np.radians(heading_deg)
        pitch = np.radians(pitch_deg)
        roll  = np.radians(roll_deg)

        cy = np.cos(yaw   * 0.5)
        sy = np.sin(yaw   * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll  * 0.5)
        sr = np.sin(roll  * 0.5)

        # 以下はスカラー同士の計算と同じ式をベクトル演算で行う
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.column_stack([w, x, y, z])  # shape (N,4)


    def rotate_vectors_by_quaternions_array(self, vecs, qs):
        """
        各行 i について、クォータニオン qs[i] (形 (4,)) で
        ベクトル vecs[i] (形 (3,)) を回転する処理をベクトル演算でまとめて実行。

        Parameters
        ----------
        vecs : np.ndarray, shape(N,3)
        qs   : np.ndarray, shape(N,4)  [w,x,y,z]

        Returns
        -------
        rotated : np.ndarray, shape(N,3)
        """
        N = len(vecs)
        vq = np.zeros((N,4))
        vq[:,1:4] = vecs

        # 逆順の回転（センサ系 → 地球系）に合わせる
        qs_conj = self.q_conjugate_array(qs)
        tmp = self.q_mult_array(qs_conj, vq)
        out_q = self.q_mult_array(tmp, qs)

        return out_q[:, 1:4]


    def q_mult_array(self, q1, q2):
        """
        クォータニオン同士の積をベクトル演算で処理。
        q1, q2 ともに shape(N,4) = [w,x,y,z]
        戻り値も shape(N,4).
        """
        w1, x1, y1, z1 = q1[:,0], q1[:,1], q1[:,2], q1[:,3]
        w2, x2, y2, z2 = q2[:,0], q2[:,1], q2[:,2], q2[:,3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2

        return np.column_stack([w,x,y,z])


    def q_conjugate_array(self, q):
        """
        クォータニオンの共役をベクトル演算で (N,4).
        q=[w,x,y,z] => conj(q)=[w,-x,-y,-z].
        """
        conj = np.copy(q)
        conj[:,1:] *= -1
        return conj



    def _lowpass_filter(self, fld):
        """
        Apply a Butterworth lowpass filter to columns [x,y,z,t].
        shape (N,4).
        """
        if len(fld) < 2:
            return fld  # not enough data to filter

        # Design filter
        b, a = signal.butter(self.filter_order, self.filter_cutoff, btype='lowpass')
        fld_filtered = np.zeros_like(fld)

        # Filter each column
        for i in range(fld.shape[1]):
            fld_filtered[:, i] = signal.filtfilt(b, a, fld[:, i])

        return fld_filtered

    def _make_dataframe_and_resample(self, df, fld_filtered):
        """
        Create a new DataFrame from original df + filtered field.
        Then resample at self.resample_rule (default 1s).
        We'll keep lat/lon from df, plus the new columns: [fldx,fldy,fldz,fldt].
        """
        # copy just lat/lon/depth for reference
        df_new = df[['Lat','Lon','Depth']].copy()
        df_new.index = df['datetime']

        # add filtered columns
        tmp = pd.DataFrame(fld_filtered, columns=['fldx','fldy','fldz','fldt'], index=df_new.index)
        df_new = pd.concat([df_new, tmp], axis=1)

        # Resample
        df_resampled = df_new.resample(self.resample_rule).bfill().interpolate(method='cubic')
        return df_resampled

    def _compute_distance(self, dj):
        """
        Compute incremental & cumulative distance based on lat/lon between consecutive rows.
        """
        if len(dj) < 2:
            dj['distance'] = 0
            dj['interval'] = 0
            return dj

        lats = dj['Lat'].values
        lons = dj['Lon'].values

        # Shifted arrays for "next" point
        lats_next = np.roll(lats, -1)
        lons_next = np.roll(lons, -1)

        # We'll compute distance row i -> i+1
        # The last row will use distance to "wrap around" the first row if we don't fix it, 
        # so let's just do length-1:
        intervals = []
        for i in range(len(dj)-1):
            dkm = distance.distance((lats[i], lons[i]), (lats_next[i], lons_next[i])).kilometers
            intervals.append(dkm)
        intervals.append(0.0)  # last row no further interval

        intervals = np.array(intervals)
        dist_cumulative = np.cumsum(intervals)

        dj['interval'] = intervals
        dj['distance'] = dist_cumulative
        return dj

    def _compute_igrf(self, dj):
        """
        Compute IGRF for each row: [igrfn, igrfe, igrfd, igrft].
        We'll store in columns of dj.
        Using ppigrf.igrf(lon, lat, alt, datetime).
        """
        if len(dj) == 0:
            for col in ['igrfn','igrfe','igrfd','igrft']:
                dj[col] = np.nan
            return dj

        # Prepare arrays
        lat_arr = dj['Lat'].values
        lon_arr = dj['Lon'].values
        # Depth => altitude? For now assume altitude = 0
        times = dj.index  # DatetimeIndex

        # Storage
        igrfn = np.zeros(len(dj))
        igrfe = np.zeros(len(dj))
        igrfd = np.zeros(len(dj))
        igrft = np.zeros(len(dj))

        for i in range(len(dj)):
            # time, lat, lon, alt=0
            dt = times[i]  # Timestamp
            lat = lat_arr[i]
            lon = lon_arr[i]

            # ppigrf.igrf(lon, lat, alt_km, datetime)
            Be, Bn, Bu = ppigrf.igrf(lon, lat, 0.0, dt)
            # Be = -float(Be)
            # Bn = -float(Bn)
            Bu = float(Bu)
            Bd = -Bu
            # store north, east, down, total
            igrfn[i] = Bn
            igrfe[i] = Be
            igrfd[i] = Bd
            igrft[i] = np.sqrt(Bn*Bn + Be*Be + Bd*Bd)

        dj['igrfn'] = igrfn
        dj['igrfe'] = igrfe
        dj['igrfd'] = igrfd
        dj['igrft'] = igrft
        return dj

    def _compute_anm(self, dj):
        """
        Compute anomaly = FLD - IGRF => [anmn, anme, anmd, anmt].
        Earth-coord FLD is [fldx, fldy, fldz, fldt].
        """
        anmn = -dj['fldx'] - dj['igrfn']
        anme = -dj['fldy'] - dj['igrfe']
        anmd = dj['fldz'] - dj['igrfd']
        anmt = dj['fldt'] - dj['igrft']

        dj['anmn'] = anmn
        dj['anme'] = anme
        dj['anmd'] = anmd
        dj['anmt'] = anmt
        return dj


# stcmkitpy/converter.py の最後に main 関数として定義
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing .stcm files")
    parser.add_argument("--coeff", default="coeff.npy")
    parser.add_argument("--intercept", default="intercept.npy")
    args = parser.parse_args()

    converter = STCMConverter(
        coeff_path=args.coeff,
        intercept_path=args.intercept
    )
    converter.process_directory(args.folder)