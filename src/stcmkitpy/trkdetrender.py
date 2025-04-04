import os
import glob
import pandas as pd
import numpy as np
from geopy.distance import geodesic


class TRKDetrender:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.trk_files = sorted(
            f for f in glob.glob(os.path.join(folder_path, "*.trk"))
            if not f.endswith(".detrended.trk")
        )

    def run(self, model='sklearn', grd_path=None):
        if not self.trk_files:
            print(f"❌ No .trk files found in: {self.folder_path}")
            return

        for file_path in self.trk_files:
            self._detrend_file(file_path, model=model, grd_path=grd_path)

        print("✔ All .trk files detrended and saved as *.detrended.trk.")

    def _detrend_file(self, file_path, model='sklearn', grd_path=None):
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None,
                             names=["time", "lon", "lat", "mag"])
            if len(df) < 2:
                print(f"  ⚠️ Skipping short file: {file_path}")
                return

            # 距離計算
            distances = [0.0]
            for i in range(1, len(df)):
                p1 = (df.loc[i-1, "lat"], df.loc[i-1, "lon"])
                p2 = (df.loc[i, "lat"], df.loc[i, "lon"])
                d = geodesic(p1, p2).kilometers
                distances.append(distances[-1] + d)
            df["dist_km"] = distances

            y = df["mag"].values

            if model == 'sklearn':
                from sklearn.linear_model import LinearRegression
                X = df["dist_km"].values.reshape(-1, 1)
                reg = LinearRegression()
                reg.fit(X, y)
                slope = reg.coef_[0]
                x0 = X[0, 0]
                trend = slope * (X.flatten() - x0)
                df["mag_detrended"] = y - trend

            elif model == 'scipy':
                from scipy.signal import detrend as scipy_detrend
                x = np.arange(len(y))
                slope, *_ = np.polyfit(x, y, deg=1)
                trend = slope * (x - x[0])
                df["mag_detrended"] = y - trend

            elif model == 'staichi_method':
                if grd_path is None:
                    raise ValueError("grd_path must be provided for staichi_method")

                import xarray as xr
                from scipy.interpolate import griddata

                lat_min, lat_max = df['lat'].min() - 1, df['lat'].max() + 1
                lon_min, lon_max = df['lon'].min() - 1, df['lon'].max() + 1

                ds = xr.open_dataset(grd_path, engine='netcdf4')

                # ↓ y方向（緯度）の順番に応じてslice方向を調整
                if ds['y'][0] > ds['y'][-1]:  # yが降順ならOK
                    subset = ds.sel(x=slice(lon_min, lon_max), y=slice(lat_max, lat_min))
                else:  # yが昇順なら順方向に
                    subset = ds.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max))

                # 安全チェック
                if subset['x'].size == 0 or subset['y'].size == 0:
                    raise ValueError("Subset grid is empty. Track may be outside the .grd range.")

                lons = subset['x'].values
                lats = subset['y'].values
                mags = subset['z'].values

                lon_grid, lat_grid = np.meshgrid(lons, lats)
                points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
                values = mags.ravel()

                query_points = df[['lon', 'lat']].values
                interpolated = griddata(points, values, query_points, method='linear')

                if np.any(np.isnan(interpolated)):
                    interpolated = griddata(points, values, query_points, method='nearest')

                # 差分から傾きを見て補正
                diff = y - interpolated
                slope, *_ = np.polyfit(df["dist_km"].values, diff, deg=1)
                trend = slope * (df["dist_km"].values - df["dist_km"].values[0])
                df["mag_detrended"] = interpolated + (diff - trend)

            else:
                raise ValueError(f"Unsupported model type: {model}")

            suffix = f".{model}.detrended.trk" if model != "scipy" else ".detrended.trk"
            output_path = file_path.replace(".trk", suffix)
            with open(output_path, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['time']:10.0f} {row['lon']:11.7f} {row['lat']:11.7f} {row['mag_detrended']:7.2f}\n")

            print(f"  - Detrended ({model}): {os.path.basename(file_path)} → {os.path.basename(output_path)}")

        except Exception as e:
            print(f"❌ Error detrending {file_path} with model={model}: {e}")
