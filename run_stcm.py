from stcmkitpy import (
    STCMPreprocessor,
    STCMBatchRegressor,
    STCMConverter,
    STCManm2trk,
    TRKDetrender,
    CrosspointFinder,
    CrossOverErrorCorrection,
)

import os
import datetime
 
'''
stcmkitpy/ ← ROOT
├── stcmkitpy/          ← PACKAGES
├── test_data/          ← TEST DATA
├── run-ascii2stcm.py   ← main run 1 (bash like format)
├── run_stcm.py         ← main run 2 (bash like format)
├── pyproject.toml
├── setup.py

'''
## 1 Prepare Calc Coeff 
fig8_folder = os.path.expanduser('~/work/stcmkitpy/KH-23-11_STCM/8fig/')
line_folder = os.path.expanduser('~/work/stcmkitpy/KH-23-11_STCM/lines/')
outname = "KH-23-11" #cruise name or project name

files = [f for f in os.listdir(fig8_folder) if f.endswith(".stcm")]

for filename in sorted(files):
    fn = os.path.join(fig8_folder, filename)
    print(f"\n=== Processing: {fn} ===")
    
    processor = STCMPreprocessor(fn=fn, bt_proton=1.0)
    processor.load_data()
    processor.compute_igrf(row_index=0)
    processor.transform_orientation_quaternion()
    processor.rpyf = processor.rpyf_quat
    processor.fit_model()
    processor.save_prep_data()
    
    print("\n--- Quaternion Transform Results ---")


## 2 Calc Coeff 
regressor = STCMBatchRegressor(fig8_folder, outname)
regressor.load_prep_files()      # read .prep files
regressor.fit_regression()       # do least-squares
regressor.print_results()        # show console output
regressor.save_coefficients()    # save coeff and intercept .npy
regressor.make_plot()            # save Plotly figure

## 3 Transform with the Coeff
# This will produce a .anm file for each .stcm in the folder.
# coeff_fn, intercept_fn = regressor.get_output_paths()
# converter = STCMConverter(coeff_fn,
#                                 intercept_fn,
#                                 filter_order=5,
#                                 filter_cutoff=0.008,
#                                 fs=10,
#                                 resample_rule='1s')

# converter.process_directory(line_folder)

##detrend

## 4 Cross over error correction
"""
1. Collection of .trk files ← Raw input data  
        ↓  
2. CrosspointFinder.find(line_folder)  
     → Detects intersection points (where tracks cross, comparing mag values)  
     → → Output: intersections.csv ← ★ This becomes the input for correction  
        ↓  
3. CrossOverErrorCorrection.run()  
     → Loads intersection_csv="intersections.csv"  
     → Estimates correction offsets for each .trk file  
     → Output: corrected_trk_allpairs/ with corrected .trk files



"""
# Convert .anm to .trk
converter = STCManm2trk(line_folder)
converter.run()

# # Detrend with sklearn or Scipy
# detrender = TRKDetrender(line_folder)
# detrender.run(model='sklearn')

# Detrend with taichi sato method
grd_file = '~/work/stcmkitpy/grd/MagAnomalyEastSoutheastAsia3rd2m.grd'
detrender = TRKDetrender(line_folder)
detrender.run(model='staichi_method', grd_path=grd_file)


# # Detrend with scipy
# detrender.run(model='scipy')

# # # 1. CrosspointFinder
# prefer_detrended_trk = True  # if use .detrended.trk files >True

# # 
# trk_pattern = resolve_trk_pattern(line_folder, prefer_detrended_trk)
# print(f"Using file pattern: {trk_pattern}")

# # 1. CrosspointFinder
# finder = CrosspointFinder(rdp_epsilon=10.0, threshold=20.0)
# intersection = finder.find(line_folder, trk_pattern=trk_pattern)

# if intersection:
#     import pandas as pd
#     intersection_clean = intersection.replace(".csv", "_clean.csv")
#     df = pd.read_csv(intersection)
#     df_clean = df.dropna(subset=["mag1", "mag2"])

#     if df_clean.empty:
#         print("❌ No valid intersection data after removing NaNs. Aborting.")
#     else:
#         df_clean.to_csv(intersection_clean, index=False)
#         print(f"✔ Cleaned CSV saved: {intersection_clean}")

#         # 2. CrossOverErrorCorrection
#         today = datetime.date.today().strftime("%Y%m%d")
#         output_dir = os.path.join(line_folder, f"corrected_trk_{today}")

#         coec = CrossOverErrorCorrection(
#             reference_file="G_SW24line07.trk",
#             intersection_csv=intersection_clean,
#             trk_dir=line_folder,
#             output_dir=output_dir,
#             trk_pattern=trk_pattern
#         )
#         coec.run()
#         print("Offset results:", coec.get_offsets())
# else:
#     print("❌ Intersection CSV could not be created. Aborting.")
