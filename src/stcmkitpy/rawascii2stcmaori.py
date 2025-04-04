import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class rawAscii2STCM_AORI:
    def __init__(self, filepath, flag_millisec=1,  nominalFs=1/10, show_fig=False, delay_ms=0):
        self.filepath = filepath
        self.flag_millisec = flag_millisec
        self.show_fig = show_fig
        self.delay_ms = delay_ms
        self.nominalFs = nominalFs
        self.basename = os.path.basename(filepath)
        self.dirname = os.path.dirname(filepath)
        self.filename_wo_ext = os.path.splitext(self.basename)[0]

    def convert(self):
        self._read_and_clean()
        self._load_dataframe()
        self._supply_millisec()
        self._resample()
        self._export_files()

    def _read_and_clean(self):
        with open(self.filepath, 'r') as f:
            first_line = f.readline()
            has_header = any(key.lower() in first_line.lower() for key in ["year", "month", "day", "hour", "lat", "lon"])

            f.seek(0)
            text = f.read()

        text = re.sub('[,:/]', '\t', text)
        tbl = str.maketrans('+', ' ', ' []')
        text = text.translate(tbl)

        if has_header:
            print("⚠️ Header line skipped")
            text = '\n'.join(text.splitlines()[1:])

        self.tempfile = os.path.join(self.dirname, f"{self.filename_wo_ext}temp.dat")
        with open(self.tempfile, 'w') as f:
            f.write(text)


    def _load_dataframe(self):
        df = pd.read_csv(self.tempfile, header=None, sep='\t', engine='python')
        cols = [0, 1, 2, 3, 4, 5, 6, 7, 9, 20, 20, 16, 17, 18, 15, 12, 13]
        colnames = ['Year', 'Month', 'Day', 'Hour','Min','Sec', 'Lat', 'Lon', 'Depth', 
                    'Proton','Gravity','Hx','Hy','Hz','Heading','Roll', 'Pitch']
        df = df.iloc[:, cols]
        df.columns = colnames

        # Gravity
        df['Gravity'] = 9999

        # Proton: no values > 99999
        df['Proton'] = pd.to_numeric(df['Proton'], errors='coerce').fillna(99999).astype(int)

        # Log: check if any actual Proton measurements are present
        if (df['Proton'] != 99999).any():
            print("🔬 Proton measurements detected in the file.")
        else:
            print("⚠️ No valid Proton data found (all values are 99999).")

        os.remove(self.tempfile)
        self.df = df



    def _supply_millisec(self):
        if self.flag_millisec != 0:
            print("⚠️ skip supply milisec")
            self.df_time = self.df.copy()
            return
        
        print("supplying millisec...")
        df = self.df.copy()
        tr = pd.to_datetime({
            'year': df.Year,
            'month': df.Month,
            'day': df.Day,
            'hour': df.Hour,
            'minute': df.Min,
            'second': df.Sec
        })
        epoch_sec = (tr - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        epoch_sec_array = epoch_sec.values

        b = np.zeros(len(epoch_sec_array)) 
        c = np.zeros(int(max(epoch_sec_array)) - int(min(epoch_sec_array) + 1))

        # 秒groupe in sec
        for i in range(int(min(epoch_sec_array) + 1), int(max(epoch_sec_array))):
            condition = (epoch_sec_array == i)
            kkk = np.extract(condition, epoch_sec_array)
            c[i - int(min(epoch_sec_array) + 1)] = len(kkk)

            if len(kkk) == 0:
                continue

            Fs = 1 / len(kkk)
            wi = np.arange(0, 1, Fs)
            pos = list(locate(epoch_sec_array, lambda x: x == i))

            for l, p in enumerate(pos):
                b[p] = epoch_sec_array[p] + wi[l]

        i = int(min(epoch_sec_array))
        condition = (epoch_sec_array == i)
        kkk = np.extract(condition, epoch_sec_array)

        Fs = 1 / c.mean() if c.mean() != 0 else 1 / self.nominalFs
        wi = np.arange(0, 1, Fs)
        geta = int(c.mean() - len(kkk))
        if geta > 1:
            wi = np.delete(wi, geta - 1)
        for l in range(len(kkk)):
            b[l] = epoch_sec_array[l] + wi[l]

        i = int(max(epoch_sec_array))
        condition = (epoch_sec_array == i)
        kkk = np.extract(condition, epoch_sec_array)

        wi = np.arange(0, 1, Fs)
        geta = int(c.mean() - len(kkk))
        if geta > 1:
            wi = np.delete(wi, -geta)
        pos = list(locate(epoch_sec_array, lambda x: x == i))
        for l, p in enumerate(pos):
            b[p] = epoch_sec_array[p] + wi[l]

        millisec_offset = np.around(b - epoch_sec_array, decimals=3)
        df['Sec'] = df.Sec + millisec_offset

        self.df_time = df.copy()

    def _resample(self):
        ddf = self.df_time.copy()

        # datetime index
        tr = pd.to_datetime({
            'year': ddf['Year'].astype(int),
            'month': ddf['Month'].astype(int),
            'day': ddf['Day'].astype(int),
            'hour': ddf['Hour'].astype(int),
            'minute': ddf['Min'].astype(int),
            'second': ddf['Sec'].astype(float)
        })
        ddf.index = tr

        # Remove Duplication
        if ddf.index.duplicated().any():
            print("⚠️ Duplication deleted")
            ddf = ddf[~ddf.index.duplicated(keep='last')]

        # heading resampling 
        ddf['A'] = np.where(ddf.Heading > 180, 1, 0)
        ddf['B'] = np.cos(np.radians(ddf.Heading))

        ddf3 = ddf.resample('25ms').bfill().interpolate(method='cubic')
        ddf4 = ddf3.asfreq(f'{self.nominalFs}s')  # 'S' → 's'


        ddf4['Heading'] = np.where(
            round(ddf4['A']) == 1,
            360 - np.degrees(np.arccos(ddf4['B'])),
            np.degrees(np.arccos(ddf4['B']))
        )

        ddf4.index = ddf4.index.shift(self.delay_ms, freq='ms')

        # format
        ddf5 = ddf4.round({
            'Sec': 3, 'Lat': 8, 'Lon': 8, 'Depth': 2,
            'Hx': 2, 'Hy': 2, 'Hz': 2, 'Roll': 2, 'Pitch': 2, 'Heading': 2
        })


        ddf5['Year'] = ddf5.index.year
        ddf5['Month'] = ddf5.index.month
        ddf5['Day'] = ddf5.index.day
        ddf5['Hour'] = ddf5.index.hour
        ddf5['Min'] = ddf5.index.minute
        ddf5['Sec'] = ddf5.index.second + ddf5.index.microsecond / 1e6

        self.df_resampled = ddf5

        if self.show_fig:
            self._plot(ddf, ddf5)




    def _plot(self, original_df, resampled_df):
        variables = ['Depth', 'Proton', 'Hx', 'Hy', 'Hz', 'Heading', 'Roll', 'Pitch']
        num_vars = len(variables)

        fig = make_subplots(
            rows=num_vars + 1, cols=1,
            shared_xaxes=False,
            vertical_spacing=0.02,
            row_heights=[1.0] + [0.25] * num_vars,
            subplot_titles=['Index Map (Lon vs Lat)'] + variables
        )

        # 1. Index Map
        fig.add_trace(
            go.Scatter(
                x=resampled_df['Lon'],
                y=resampled_df['Lat'],
                mode='markers+lines',
                name='Trajectory',
                marker=dict(size=4),
                text=resampled_df.index.astype(str),
                hovertemplate="Time: %{text}<br>Lon: %{x:.5f}<br>Lat: %{y:.5f}<extra></extra>"
            ),
            row=1, col=1
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

        # 2. Other variables
        for i, var in enumerate(variables, start=2):
            fig.add_trace(
                go.Scatter(
                    x=resampled_df.index,
                    y=resampled_df[var],
                    mode='lines',
                    name=f'{var} (resampled)',
                    line=dict(width=1)
                ),
                row=i, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=original_df.index,
                    y=original_df[var],
                    mode='lines',
                    name=f'{var} (original)',
                    line=dict(width=1, dash='dot'),
                    opacity=0.4
                ),
                row=i, col=1
            )

        # 3. Layout
        fig.update_layout(
            height=250 * (num_vars + 1),
            width=1000,
            title_text=f'Sensor Time Series with Index Map: {os.path.basename(self.filepath)}',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='top',
                y=-0.02,  # 👈 タイトルと重ならないよう下に配置
                xanchor='center',
                x=0.5
            ),
            margin=dict(t=80, b=60)  # 👈 余白を少し広げると良
        )


        # 4. Save
        out_html = self.filepath + ".html"
        fig.write_html(out_html)
        print(f"✅ sensor + index map plot saved to: {out_html}")



    def _export_files(self):
        df = self.df_resampled.copy()

        cols = ['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec',
                'Lat', 'Lon', 'Depth', 'Proton', 'Gravity',
                'Hx', 'Hy', 'Hz', 'Heading', 'Roll', 'Pitch']
        df = df[cols]

        ddd6 = df.to_numpy(float)

        # format settings
        ddd6[:, 0] -= 2000  # Year 
        ddd6[:, 6] *= 1e7   # Lat
        ddd6[:, 7] *= 1e7   # Lon
        ddd6[:, 14] *= 100  # Heading
        ddd6[:, 15] *= 100  # Roll
        ddd6[:, 16] *= 100  # Pitch

        # Export .int.stcm 
        out_int = self.filepath + ".int.stcm"
        np.savetxt(
            out_int,
            ddd6,
            fmt='%02d %02d %02d %02d %02d %07.4f %09d %09d %05d %04d %04d %05d %05d %05d %05d %03d %03d',
            delimiter=' '
        )

        # Export .float.stcm
        ddd7 = df.to_numpy(float)
        out_float = self.filepath + ".float.stcm"
        np.savetxt(
            out_float,
            ddd7,
            fmt='%04d %02d %02d %02d %02d %06.3f %012.8f %012.8f %07.2f %05d %04d %05.2f %05.2f %05.2f %06.2f %05.2f %05.2f',
            delimiter=' '
        )

        print(f"✅ saved: {out_int}")
        print(f"✅ saved: {out_float}")


        # STCMKit用オプション生成（1行だけ表示）
        # yi = int(ddd7[0, 0]) - 2000
        # mi = int(ddd7[0, 1])
        # di = int(ddd7[0, 2])
        # lati = int(ddd7[0, 6])
        # latf = (ddd7[0, 6] - lati) * 60
        # loti = int(ddd7[0, 7])
        # lotf = (ddd7[0, 7] - loti) * 60

        # print(f"📎 STCMKit cmd: calc_coeff_prep {os.path.basename(out_int)} -D{yi}/{mi}/{di} -P{lati}:{latf:.2f}/{loti}:{lotf:.2f} > temp.dat")
        # print("📎 STCMKit cmd: calc_coeff temp.dat")
        # print(f"🕒 Please check the delay at resample; now the delay is {self.delay_ms} ms")



# if __name__ == "__main__":
#     converter = AORIraw2STCMformatConverter(
#         filepath="/home/koe3/work/stcmkitpy/KH-23-11_STCM/fig8-2.dat",
#         flag_millisec=1,  # MILISEC EXIST(1) OR NOT(0)
#         flag_proton=1,    # PROTON EXIST(1) OR NOT(0)
#         show_fig=True,    # SHOW HEADING GRAPH
#         delay_ms=0        # RESAMPLING DELAY
#     )
#     converter.convert()
