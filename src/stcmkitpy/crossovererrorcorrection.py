import numpy as np
import csv
import glob
import os

class CrossOverErrorCorrection:
    def __init__(self, reference_file, intersection_csv, trk_dir=".", output_dir="corrected_trk_allpairs"):
            self.reference_file = reference_file
            self.intersection_csv = intersection_csv
            self.trk_dir = trk_dir
            self.output_dir = output_dir
            self.offsets = {}
            self.track_files = []
            self.index_map = {}
            self.A = None
            self.b = None

    def run(self):
        self._validate_inputs()
        self._load_trk_files()
        self._build_equation_system()
        self._solve_offsets()
        self._apply_corrections()

    def _validate_inputs(self):
        if not os.path.isfile(self.reference_file):
            raise FileNotFoundError(f"Reference file not found: {self.reference_file}")
        if not os.path.isfile(self.intersection_csv):
            raise FileNotFoundError(f"Intersection CSV not found: {self.intersection_csv}")

    def _load_trk_files(self):
            self.track_files = sorted(glob.glob(os.path.join(self.trk_dir, "*.trk")))
            if not self.track_files:
                raise RuntimeError(f"No .trk files found in directory: {self.trk_dir}")

            index = 0
            ref_path = os.path.join(self.trk_dir, self.reference_file)
            for fname in self.track_files:
                name = os.path.basename(fname)
                if name == self.reference_file:
                    self.index_map[fname] = None
                else:
                    self.index_map[fname] = index
                    index += 1
            print("Loaded track files:", self.track_files)
            print("Index map:", self.index_map)

    def _build_equation_system(self):
        rows = []
        b_vals = []

        with open(self.intersection_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for line_num, row in enumerate(reader, start=2):
                try:
                    file1, file2 = row[1], row[4]
                    mag1, mag2 = float(row[7]), float(row[8])
                    if np.isnan(mag1) or np.isnan(mag2):
                        raise ValueError("NaN detected")
                except (IndexError, ValueError) as e:
                    print(f"Skipping malformed line {line_num}: {e}")
                    continue

                diff = mag2 - mag1
                idx1 = self.index_map.get(file1)
                idx2 = self.index_map.get(file2)

                if (idx1 is not None and idx1 >= len(self.track_files) - 1) or \
                   (idx2 is not None and idx2 >= len(self.track_files) - 1):
                    print(f"⚠️ Skipping invalid index: {file1} → {idx1}, {file2} → {idx2}")
                    continue

                row_vec = np.zeros(len(self.track_files) - 1)

                if idx1 is None and idx2 is None:
                    continue
                elif idx1 is None:
                    row_vec[idx2] = 1.0
                    rows.append(row_vec)
                    b_vals.append(-diff)
                elif idx2 is None:
                    row_vec[idx1] = 1.0
                    rows.append(row_vec)
                    b_vals.append(diff)
                else:
                    row_vec[idx1] = 1.0
                    row_vec[idx2] = -1.0
                    rows.append(row_vec)
                    b_vals.append(diff)

        if not rows:
            raise RuntimeError("No valid intersection data found in CSV.")

        self.A = np.vstack(rows)
        self.b = np.array(b_vals)

    def _solve_offsets(self):
        O_vec, *_ = np.linalg.lstsq(self.A, self.b, rcond=None)

        for fname, idx in self.index_map.items():
            if idx is None:
                self.offsets[fname] = 0.0
            elif 0 <= idx < len(O_vec):  # for safe
                self.offsets[fname] = O_vec[idx]
            else:
                print(f"⚠️ Skipping out-of-bounds index for {fname}: idx={idx}")
                self.offsets[fname] = 0.0  # fallback


    def _apply_corrections(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for full_path in self.track_files:
            offset = self.offsets.get(full_path, 0.0)
            basename = os.path.basename(full_path)
            
            coer_name = os.path.splitext(basename)[0] + ".trk.coer"
            output_path = os.path.join(self.output_dir, coer_name)

            with open(full_path, "r", encoding="utf-8") as fin, \
                 open(output_path, "w", encoding="utf-8", newline="") as fout:

                for line_num, line in enumerate(fin, start=1):
                    line = line.strip()
                    if not line:
                        fout.write("\n")
                        continue

                    parts = line.split()
                    if len(parts) < 4:
                        print(f"Skipping malformed line {line_num} in {basename}")
                        fout.write(line + "\n")
                        continue

                    try:
                        t, lon, lat, mag = parts
                        corrected_mag = float(mag) + offset
                        fout.write(f"{t} {lon} {lat} {corrected_mag:.1f}\n")
                    except ValueError:
                        print(f"Skipping line {line_num} in {basename}: cannot parse float.")
                        fout.write(line + "\n")

        print(f"✔ Correction applied using reference: {self.reference_file}")
        print(f"✔ Corrected .trk files saved in: {self.output_dir}")

    def get_offsets(self):
        return self.offsets
