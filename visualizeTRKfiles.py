import glob
import math
import folium
import branca.colormap as cm

def main():
    # 1) .trk ファイルを全部読み込む
    trk_files = glob.glob("*.trk")
    if not trk_files:
        print("No .trk files found.")
        return
    
    all_points = []  # (lat, lon, mag, file_name, unixtime)
    
    for file_name in trk_files:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                unixtime = parts[0]
                lon = float(parts[1])
                lat = float(parts[2])
                mag = float(parts[3])
                
                all_points.append((lat, lon, mag, file_name, unixtime))
    
    if not all_points:
        print("No points found in any .trk files.")
        return
    
    # 2) mag の最小値・最大値を取得
    mags = [p[2] for p in all_points]
    min_mag = min(mags)
    max_mag = max(mags)
    
    # 3) 地図の中心 (平均座標) を決める
    avg_lat = sum(p[0] for p in all_points) / len(all_points)
    avg_lon = sum(p[1] for p in all_points) / len(all_points)
    
    # 4) Foliumマップ生成
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
    
    # 5) jet風カラーマップを定義
    jet_colormap = cm.LinearColormap(
        ['#00007F', 'blue', '#007FFF', 'cyan', 
         '#7FFF7F', 'yellow', '#FF7F00', 'red', '#7F0000'],
        vmin=min_mag,
        vmax=max_mag
    )
    jet_colormap.caption = "mag value"
    jet_colormap.add_to(m)  # 地図にカラーバーを追加
    
    # 6) 各点を CircleMarker で描画
    for (lat, lon, mag, fname, utime) in all_points:
        circle_color = jet_colormap(mag)
        tooltip_text = f"File: {fname}<br>Unixtime: {utime}<br>Mag: {mag}"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            fill=True,
            fill_color=circle_color,
            color=None,
            fill_opacity=0.8,
            tooltip=tooltip_text
        ).add_to(m)
    
    # 7) HTMLとして保存
    output_html = "map.html"
    m.save(output_html)
    print(f"Done. Open '{output_html}' in a browser to view the map.")

if __name__ == "__main__":
    main()
