import os
from stcmkitpy import rawAscii2STCM_AORI

# Target
input_dir = os.path.expanduser('~/work/stcmkitpy/KH-23-11_STCM/rawascii/')

# make list .dat
file_list = [f for f in os.listdir(input_dir) if f.endswith('.dat')]

for fname in file_list:
    filepath = os.path.join(input_dir, fname)
    print(f"📄 Processing: {fname}")

    converter = rawAscii2STCM_AORI(
        filepath=filepath,
        flag_millisec=1, # 1 > milisec data exists, 0 > no milisec (old AORI data)
        show_fig=True, # Visualize data
        delay_ms=0, # you can move the delay fix
        nominalFs=1/10 # resampling rate
    )
    converter.convert()
    print(f"✅ Done: {fname}\n")
