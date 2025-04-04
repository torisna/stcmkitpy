import pandas as pd
import time
import glob
import os

class STCManm2trk:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.anm_files = sorted(glob.glob(os.path.join(folder_path, '*.anm')))

    def run(self):
        if not self.anm_files:
            print(f"No .anm files found in: {self.folder_path}")
            return

        for file_path in self.anm_files:
            self._convert_file(file_path)

        print("✔ All .anm files converted to .trk format.")

    def _convert_file(self, file_path):
        try:
            # Read file
            df = pd.read_csv(file_path)

            # Extract data（datetime → UNIXtime, lon, lat, anmt）
            df = df[['datetime', 'Lon', 'Lat', 'anmt']].copy()
            df['time'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df['time'] = df['time'].apply(lambda x: int(x.timestamp()))

            df = df[['time', 'Lon', 'Lat', 'anmt']]
            df.columns = ['time', 'lon', 'lat', 'mag']

            # Output file name (.trk)
            output_file = os.path.splitext(file_path)[0] + '.trk'

            with open(output_file, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['time']:10.0f} {row['lon']:11.7f} {row['lat']:11.7f} {row['mag']:6.1f}\n")

            print(f"  - Converted: {os.path.basename(file_path)} → {os.path.basename(output_file)}")

        except Exception as e:
            print(f"❌ Failed to convert {file_path}: {e}")
