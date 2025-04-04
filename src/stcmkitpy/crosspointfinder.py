import os
import math
import glob
import datetime

class CrosspointFinder:
    def __init__(self, rdp_epsilon=10.0, threshold=20.0):
        self.RDP_EPSILON = rdp_epsilon
        self.THRESHOLD = threshold
        self.ORIGIN_LAT = None
        self.ORIGIN_LON = None
        self.tracks = []
        self.results = []

    def find(self, folder_path):
        trk_files = sorted(glob.glob(os.path.join(folder_path, "*.trk")))
        if not trk_files:
            print(f"⚠ No .trk files found in: {folder_path}")
            return None  # ←変更

        center = self._calc_center_latlon(trk_files)
        if center is None:
            print("⚠ No valid coordinates found in .trk files.")
            return None  # ←変更

        self.ORIGIN_LAT, self.ORIGIN_LON = center
        print(f"✔ Center lat/lon = ({self.ORIGIN_LAT:.6f}, {self.ORIGIN_LON:.6f})")

        self._load_tracks(trk_files)
        self._detect_intersections()

        # 出力ファイル名に日付をつける
        today = datetime.date.today().strftime("%Y%m%d")
        out_csv = os.path.join(folder_path, f"intersections_{today}.csv")
        return self._write_results(output_path=out_csv) 

    def _calc_center_latlon(self, trk_files):
        total_lat, total_lon, count = 0.0, 0.0, 0
        for file_name in trk_files:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        try:
                            lon = float(parts[1])
                            lat = float(parts[2])
                            total_lat += lat
                            total_lon += lon
                            count += 1
                        except ValueError:
                            continue
            except OSError as e:
                print(f"❌ Failed to read {file_name}: {e}")

        return (total_lat / count, total_lon / count) if count > 0 else None

    def _latlon_to_xy(self, lat, lon):
        R = 6378137.0
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        o_lat_rad = math.radians(self.ORIGIN_LAT)
        o_lon_rad = math.radians(self.ORIGIN_LON)
        x = (lon_rad - o_lon_rad) * math.cos((lat_rad + o_lat_rad) / 2) * R
        y = (lat_rad - o_lat_rad) * R
        return x, y

    def _rdp(self, points, epsilon):
        if len(points) < 3:
            return points
        first, last = points[0], points[-1]
        max_dist, max_index = -1, 0
        for i in range(1, len(points) - 1):
            dist = self._perpendicular_distance(points[i], first, last)
            if dist > max_dist:
                max_dist = dist
                max_index = i
        if max_dist > epsilon:
            left = self._rdp(points[:max_index + 1], epsilon)
            right = self._rdp(points[max_index:], epsilon)
            return left[:-1] + right
        else:
            return [first, last]

    def _perpendicular_distance(self, point, start, end):
        (x, y), (x1, y1), (x2, y2) = point, start, end
        if (x1 == x2 and y1 == y2):
            return math.hypot(x - x1, y - y1)
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        den = math.hypot(y2 - y1, x2 - x1)
        return num / den

    def _simplify_to_segments(self, points, epsilon):
        just_xy = [(p[0], p[1]) for p in points]
        simplified = self._rdp(just_xy, epsilon)
        if len(simplified) <= 2:
            return [(simplified[0], simplified[-1])]
        return [(simplified[i], simplified[i + 1]) for i in range(len(simplified) - 1)]

    def _load_tracks(self, trk_files):
        self.tracks = []
        for fname in trk_files:
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    points = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue
                        try:
                            t = float(parts[0])
                            lon = float(parts[1])
                            lat = float(parts[2])
                            mag = float(parts[3])
                        except ValueError:
                            continue
                        x, y = self._latlon_to_xy(lat, lon)
                        points.append((x, y, lat, lon, mag, t))
                if len(points) < 2:
                    continue
                segments = self._simplify_to_segments(points, self.RDP_EPSILON)
                self.tracks.append({
                    "file_name": fname,
                    "points": points,
                    "segments": segments
                })
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")

    def _segments_intersect(self, seg1, seg2):
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < 1e-12:
            return False, None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            xi = x1 + ua * (x2 - x1)
            yi = y1 + ua * (y2 - y1)
            return True, (xi, yi)
        return False, None

    def _find_near_points(self, track, xy, threshold):
        near = []
        for (x, y, lat, lon, mag, t) in track["points"]:
            dx, dy = x - xy[0], y - xy[1]
            if abs(dx) > threshold or abs(dy) > threshold:
                continue
            dist = math.hypot(dx, dy)
            if dist <= threshold:
                near.append({
                    "file_name": track["file_name"],
                    "x": x, "y": y,
                    "lat": lat, "lon": lon,
                    "mag": mag, "time": t,
                    "dist": dist
                })
        return near

    def _detect_intersections(self):
        self.results = []
        id_counter = 1
        total = len(self.tracks)
        for i in range(total):
            for j in range(i + 1, total):
                for segA in self.tracks[i]["segments"]:
                    for segB in self.tracks[j]["segments"]:
                        is_cross, xy_cross = self._segments_intersect(segA, segB)
                        if is_cross:
                            near_i = self._find_near_points(self.tracks[i], xy_cross, self.THRESHOLD)
                            near_j = self._find_near_points(self.tracks[j], xy_cross, self.THRESHOLD)
                            if near_i and near_j:
                                best_pair = min(
                                    ((p1, p2) for p1 in near_i for p2 in near_j),
                                    key=lambda pair: pair[0]["dist"] + pair[1]["dist"],
                                    default=None
                                )
                                if best_pair:
                                    p1, p2 = best_pair
                                    self.results.append((
                                        id_counter,
                                        p1["file_name"], p1["lat"], p1["lon"],
                                        p2["file_name"], p2["lat"], p2["lon"],
                                        p1["mag"], p2["mag"]
                                    ))
                                    id_counter += 1

    def _write_results(self, output_path="intersections.csv"):
        try:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                f.write("intersection_id,file_name_1,lat_1,lon_1,file_name_2,lat_2,lon_2,mag1,mag2\n")
                for row in self.results:
                    f.write(",".join(map(str, row)) + "\n")
            print(f"✔ Results saved to {output_path}")
            return os.path.abspath(output_path)  # ★これを追加！
        except Exception as e:
            print(f"❌ Failed to write output CSV: {e}")
            return None

