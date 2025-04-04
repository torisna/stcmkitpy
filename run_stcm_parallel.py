import os
import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from stcmkitpy import (
    STCMPreprocessor,
    STCMBatchRegressor,
    STCMConverter,
    STCManm2trk,
    CrosspointFinder,
    CrossOverErrorCorrection,
)

# ## 1 Prepare Calc Coeff 
folder = "/home/koe3/work/stcmkitpy/test_data2/8figures"

files = [f for f in os.listdir(folder) if f.endswith(".stcm")]

for filename in sorted(files):
    fn = os.path.join(folder, filename)
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
# outname = "G_SW24"
outname = "GB23"

regressor = STCMBatchRegressor(folder, outname)
regressor.load_prep_files()      # read .prep files
regressor.fit_regression()       # do least-squares
regressor.print_results()        # show console output
regressor.save_coefficients()    # save coeff and intercept .npy
regressor.make_plot()            # save Plotly figure

## 3 Transform with the Coeff
line_folder = "/home/koe3/work/stcmkitpy/test_data2/lines"

coeff_fn, intercept_fn = regressor.get_output_paths()
converter = STCMConverter(coeff_fn, intercept_fn)
stcm_files = sorted([
    os.path.join(line_folder, f)
    for f in os.listdir(line_folder)
    if f.endswith('.stcm')
])
if not stcm_files:
    print("No .stcm files found.")
    exit()
print(f"🛠 Converting {len(stcm_files)} files in parallel...")

max_workers = max(1, multiprocessing.cpu_count() - 1)
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(converter.process_single_file, fn) for fn in stcm_files]

    for i, future in enumerate(futures):
        try:
            future.result()  # Error
        except Exception as e:
            print(f"❌ Error in file {stcm_files[i]}: {e}")

print("✅ All files processed.")

## detrend

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

 ---
test_data/lines/
├── G_SW24line01.trk
├── G_SW24line07.trk
├── intersections.csv
└── corrected_trk_20250402/  
    ├── G_SW24line01.trk
    ├── G_SW24line07.trk
    └── ...
 

"""
# 0. Convert .anm to .trk
converter = STCManm2trk(line_folder)
converter.run()

# 1. CrosspointFinder
finder = CrosspointFinder(rdp_epsilon=10.0, threshold=20.0)
intersection = finder.find(line_folder)

if intersection:
    # 2. CrossOverErrorCorrection
    today = datetime.date.today().strftime("%Y%m%d")
    output_dir = os.path.join(line_folder, f"corrected_trk_{today}")

    coec = CrossOverErrorCorrection(
        reference_file="/home/koe3/work/stcmkitpy/test_data2/lines/G_SW24line07.trk",
        intersection_csv=intersection,
        trk_dir=line_folder,
        output_dir=output_dir
    )
    coec.run()
else:
    print("❌ Intersection CSV could not be created. Aborting.")

print("Offset results:", coec.get_offsets())