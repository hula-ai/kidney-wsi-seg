# Iterate over different slide ids of the processed_list.csv (dataframe) (line 652)

## Deleting old prediction masks/multi/tiles (line 668)

## Loading the current slide id based on their format/extension (line 696 - 853)

## Instantiating our pixel conf/counts np arrays ... (line 862)

## Generating GT masks (for each compartment class) if we have them ... (line 865)

## Find bounding box and contour area of each contour (extracting contours (each contour is a list of coordinates)) (line 233-234)

## Extracting patches from each contour (-> The number of patches is greater than the number of contours) (line 1051)

### Check if small edge instance found -> skipping (line 1134)

## Patch-wise prediction complete (line 1158)

## a loop - Generating our cutoff thresholds (line 1202)

## a loop (line 1299)
## a loop (line 1389)
## a loop (line 1442)
## a loop (line 1457)
## a loop (line 1477)
## a loop (line 1489)
## a loop (line 1530)
## a loop (line 1578)

## Bug (line 1635 - 1646)

1/ turn + into - an - into + 

```python
if img_x < 0:
    img_x_end += img_x
    img_x = 0
if img_y < 0:
    img_y_end += img_y
    img_y = 0
if img_x_end > wsi.shape[1]:
    img_x += (img_x_end - wsi.shape[1])
    img_x_end = wsi.shape[1]
if img_y_end > wsi.shape[0]:
    img_y += (img_y_end - wsi.shape[0])
    img_y_end = wsi.shape[0]
```