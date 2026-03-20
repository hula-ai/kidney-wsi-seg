#### `generate_wsi_performance_figures_tables.py`

- INPUT: csv files produced by running `generate_wsi_prediction_masks.py`

```python
DATAFRAME_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_TMA.csv'
```

- OUTPUT: seaborn figure for PR curves, threshold comparison tables, other metrics tables

```python
RESULTS_SAVE_DIR = '/data/public/HULA/WSIs_renal_compartment_segmentations_new2/'
```

- classes:

```python
CLASSES = ["Glomerulus", "Arteriole", "Artery"]
```


#### `pangn_performance.py`

- INPUT: csv files produced by running `generate_wsi_prediction_masks.py`

```python
DATAFRAME_DIR = ['/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Cologne_3.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Szeged_6.csv',
'/data/public/HULA/WSIs_renal_compartment_segmentations/process_list_Pan_GN_Bari_1.csv',
```

```python
class_info = pd.read_csv('/data/public/HULA/Pan_GN/Pan_GN_GT.csv')
```

- OUTPUT: precision, recall, mAUC, classwise PR curve, all PR curve
'Classes', 'mIOU', 'mAP', 'mAR', 'mF1', 'mAS'

```python
RESULTS_SAVE_DIR = './pangn_seg_results/2/'
```

- `classes` and `disease classes`:

```python
CLASSES = ["Glomerulus"]

DISEASE_CLASSES = ["ABMGN", "ANCA", "C3-GN", "CryoglobulinemicGN", "DDD", "Fibrillary", "IAGN",
                   "IgAGN", "Membranous", "MPGN", "PGNMID", "SLEGN-IV"]
```

#### `json_test_confidence_clustering.py`

- "This script does clustering on the pseudo annotations predicted by the teacher network. This will find the ideal
cutoffs for each class (Background, Glomerulus, Arteriole, Artery), which are used in other scripts to filter out noisy pseudo labels."

- INPUT:

```python
mmdet_segm_test_path = "/data/syed/mmdet/results/run19_ep2_25k_json_results.segm.json"
```

- OUTPUT:

```python
print(cutoffs)
```

#### `visualize_log_file.py`

- INPUT: 

```python
RESULTS_DIR = '/data/syed/mmdet/run20_swin_gan_img_train_iter2_autoaug/'
```

- OUTPUT: png files are saved in `RESULTS_DIR`

Bounding box validation performance (precision)

Segmentation validation performance (precision)

4 dataframes: bbox_val_AP, class_bbox_val_AP, seg_val_AP, class_segm_val_AP