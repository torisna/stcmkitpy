import glob
import os

class TrkLeveler:
    def __init__(self, delta_mag=0.0):
        """
        Parameters:
            delta_mag (float): Magnetic offset to apply (in nT).
                               Example: 224.541 for leveling to STCM
        """
        self.delta_mag = delta_mag

    def set_delta_mag(self, value):
        """Update the delta_mag if needed later"""
        self.delta_mag = value

    def level_all(self, pattern="*.anm.trk"):
        """Process all matching .trk files in current directory"""
        trk_files = glob.glob(pattern)
        if not trk_files:
            print("No .anm.trk files found.")
            return
        
        for trk_file in trk_files:
            self._level_single(trk_file)
            print(f"Done: {trk_file} -> {trk_file}.leveled.trk")

    def _level_single(self, trk_file):
        output_file = trk_file + ".leveled.trk"
        with open(trk_file, "r", encoding="utf-8") as fin, \
             open(output_file, "w", encoding="utf-8", newline="") as fout:

            for line in fin:
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue

                parts = line.split()
                if len(parts) < 4:
                    fout.write(line + "\n")
                    continue

                time_val = float(parts[0])
                lon_val  = float(parts[1])
                lat_val  = float(parts[2])
                mag_val  = float(parts[3])

                # Apply delta_mag
                mag_val += self.delta_mag

                # Output formatted line
                out_line = (
                    f"{time_val:10.0f}"
                    f"{lon_val:11.7f}"
                    f"{lat_val:11.7f}"
                    f"{mag_val:6.1f}"
                )
                fout.write(out_line + "\n")
