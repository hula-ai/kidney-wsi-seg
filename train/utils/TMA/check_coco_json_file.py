import os
import json
import pandas as pd


data_path = '/data/syed'

train_path = os.path.join(data_path, 'coco_tma_tile_train_folds_1-4.json')

val_path = os.path.join(data_path, 'coco_tma_tile_validation_fold0_fixed.json')

# test_path = os.path.join(data_path, 'coco_tma_tile_validation_fold0_fixed.json')


# f = open(train_path)
with open(train_path) as f:
  data = json.load(f)
with open(val_path) as f:
  data_val = json.load(f)
for key, value in data.items():
  print('[*] key: ', key, ', type: ', type(value), ', len(value): ', len(value))
# [*] key:  info
# [*] key:  images
# [*] key:  annotations
# [*] key:  categories

print('===== Images =====')
# Assign cls labels
dict_of_mapping_from_caseid_to_label = {}
tma_csv_path = '/data/public/HULA/TMA/TMA_AICohort.csv'
tma_csv_df = pd.read_csv(tma_csv_path)
biopsy_number_list = tma_csv_df['Biopsy Number'].tolist()
case_label_list = tma_csv_df['Case Label'].tolist()

for idx, biopsy_number in enumerate(biopsy_number_list):
  if biopsy_number in ['C 19.1807', 'C 21.4974']: # 2 case ids that have trailing 0 in their names in the folder multi
    biopsy_number = biopsy_number + '0' 
  dict_of_mapping_from_caseid_to_label[biopsy_number] = case_label_list[idx]

train_list_of_uniq_slide_ids = []
train_list_of_uniq_case_ids = []
train_list_of_uniq_labels = []
for image in data['images']:
  train_list_of_uniq_slide_ids.append(image['file_name'].split(' [')[0].rsplit('/', 1)[-1])
train_list_of_uniq_slide_ids = list(set(train_list_of_uniq_slide_ids))
train_list_of_uniq_slide_ids.sort()
print('- Train slide_id: ', train_list_of_uniq_slide_ids[:4])
print('- Number of train slide id: ', len(train_list_of_uniq_slide_ids))
# --- case id ---
for slide_id in train_list_of_uniq_slide_ids:
  if slide_id[0] == 'C':
    tmp1, tmp2 = slide_id.split(' ', 2)[0], slide_id.split(' ', 2)[1]
    train_list_of_uniq_case_ids.append(''.join([tmp1, ' ', tmp2]))
  elif slide_id[:2] == 'NC':
    if slide_id[2] == ' ':
      tmp1, tmp2 = slide_id.split(' ', 2)[0], slide_id.split(' ', 2)[1]
      train_list_of_uniq_case_ids.append(''.join([tmp1, ' ', tmp2]))
    else:
      tmp = slide_id.split(' ', 2)[0]
      train_list_of_uniq_case_ids.append(tmp)
  elif slide_id[:3] == 'WCM':
    train_list_of_uniq_case_ids.append(slide_id.rsplit('_', 1)[0])
  else:
    print('!!! Other case ids in train: ', slide_id)
# --- label ---
for case_id in train_list_of_uniq_case_ids:
  if case_id in dict_of_mapping_from_caseid_to_label:
    train_list_of_uniq_labels.append(dict_of_mapping_from_caseid_to_label[case_id])
  else:
    train_list_of_uniq_labels.append('None')

val_list_of_uniq_slide_ids = []
val_list_of_uniq_case_ids = []
val_list_of_uniq_labels = []
for image in data_val['images']:
  val_list_of_uniq_slide_ids.append(image['file_name'].split(' [')[0].rsplit('/', 1)[-1])
val_list_of_uniq_slide_ids = list(set(val_list_of_uniq_slide_ids))
val_list_of_uniq_slide_ids.sort()
print('- Val slide_id: ', val_list_of_uniq_slide_ids[:4])
print('- Number of val slide id: ', len(val_list_of_uniq_slide_ids))
# --- case id ---
for slide_id in val_list_of_uniq_slide_ids:
  if slide_id[0] == 'C':
    tmp1, tmp2 = slide_id.split(' ', 2)[0], slide_id.split(' ', 2)[1]
    val_list_of_uniq_case_ids.append(''.join([tmp1, ' ', tmp2]))
  elif slide_id[:2] == 'NC':
    if slide_id[2] == ' ':
      tmp1, tmp2 = slide_id.split(' ', 2)[0], slide_id.split(' ', 2)[1]
      val_list_of_uniq_case_ids.append(''.join([tmp1, ' ', tmp2]))
    else:
      tmp = slide_id.split(' ', 2)[0]
      val_list_of_uniq_case_ids.append(tmp)
  elif slide_id[:3] == 'WCM':
    val_list_of_uniq_case_ids.append(slide_id.rsplit('_', 1)[0])
  else:
    print('!!! Other case ids in val: ', slide_id)
# --- label ---
for case_id in val_list_of_uniq_case_ids:
  if case_id in dict_of_mapping_from_caseid_to_label:
    val_list_of_uniq_labels.append(dict_of_mapping_from_caseid_to_label[case_id])
  else:
    val_list_of_uniq_labels.append('None')

# Check if any train slide id in val slide id, or if any val slide id in train slide id
# for train_slide_id in train_list_of_uniq_slide_ids:
#   if train_slide_id in val_list_of_uniq_slide_ids:
#     print('!! train in val: ', train_slide_id)
# for val_slide_id in val_list_of_uniq_slide_ids:
#   if val_slide_id in train_list_of_uniq_slide_ids:
#     print('!! val in train: ', val_slide_id)

train_ret_dict = {'train_slide_ids': train_list_of_uniq_slide_ids, 'train_case_ids': train_list_of_uniq_case_ids, 'train_labels': train_list_of_uniq_labels} 
train_ret_df = pd.DataFrame.from_dict(train_ret_dict, orient='index')
train_ret_df = train_ret_df.transpose()
print('==> train_list_of_uniq_case_ids', len(list(set(train_list_of_uniq_case_ids))))
tma_cnt = 0
mimicker_cnt = 0
for case_id in list(set(train_list_of_uniq_case_ids)):
  if case_id not in dict_of_mapping_from_caseid_to_label:
    continue
  if dict_of_mapping_from_caseid_to_label[case_id] == 'TMA':
    tma_cnt += 1
  elif dict_of_mapping_from_caseid_to_label[case_id] == 'Mimicker':
    mimicker_cnt += 1
print('==> In train: {} TMAs, {} Mimickers'.format(tma_cnt, mimicker_cnt))

# ret_df.to_csv('/data/public/HULA/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks/coco_tma_tile_old_slideids_3classes.csv')
val_ret_dict = {'val_slide_ids': val_list_of_uniq_slide_ids, 'val_case_ids': val_list_of_uniq_case_ids, 'val_labels': val_list_of_uniq_labels} 
val_ret_df = pd.DataFrame.from_dict(val_ret_dict, orient='index')
val_ret_df = val_ret_df.transpose()
print('==> val_list_of_uniq_case_ids', len(list(set(val_list_of_uniq_case_ids))))
tma_cnt = 0
mimicker_cnt = 0
for case_id in list(set(val_list_of_uniq_case_ids)):
  if case_id not in dict_of_mapping_from_caseid_to_label:
    continue
  if dict_of_mapping_from_caseid_to_label[case_id] == 'TMA':
    tma_cnt += 1
  elif dict_of_mapping_from_caseid_to_label[case_id] == 'Mimicker':
    mimicker_cnt += 1
print('==> In val: {} TMAs, {} Mimickers'.format(tma_cnt, mimicker_cnt))
# print(train_ret_df)
# print(val_ret_df)

writer = pd.ExcelWriter('/data/public/HULA/TMA/QuPath_Projects/TMASegmentationR14/tiles_and_tile_masks/coco_tma_tile_old_slideids_caseids_labels_3classes.xlsx', engine='xlsxwriter')
workbook = writer.book
worksheet1 = workbook.add_worksheet('train')
writer.sheets['train'] = worksheet1
worksheet2 = workbook.add_worksheet('val')
writer.sheets['val'] = worksheet2

# For break line (Alt + Enter) in each cell
wrap_format = workbook.add_format({'text_wrap': True})
wrap_format.set_align('center')
wrap_format.set_align('vcenter')
worksheet1.set_column('A:Z', 12, wrap_format)
worksheet2.set_column('A:Z', 12, wrap_format)

# _dataframe.to_excel(writer, sheet_name='AllMetrics')
train_ret_df.to_excel(writer, sheet_name='train')
val_ret_df.to_excel(writer, sheet_name='val')
writer.save()


print('===== Annotation =====')
print('--- Train ---')
stat_id = []
stat_imageid = []
stat_categoryid = []
stat_len_seg = []
stat_iscrowd = []
stat_bbox = []
for idx, annot in enumerate(data['annotations']):
  stat_id.append(annot['id'])
  stat_imageid.append(annot['image_id'])
  stat_categoryid.append(annot['category_id'])
  stat_len_seg.append(len(annot['segmentation']))
  stat_iscrowd.append(annot['iscrowd'])
  stat_bbox.append(len(annot['bbox']))
  if idx == 0:
    print('- first bbox: ', annot['bbox'])
    print('- first area: ', annot['area'])

print('Unique ids: ', len(list(set(stat_id))))
print('Unique image ids: ', len(list(set(stat_imageid))))
print('Unique category ids: ', set(stat_categoryid))
print('Unique lens of segmentation: ', set(stat_len_seg))
print('Unique iscrowd: ', set(stat_iscrowd))
print('Unique lens of bbox', set(stat_bbox))

counting_each_categoryid = [0]*4
for categoryid in stat_categoryid:
  counting_each_categoryid[categoryid] += 1
print('=> Count each category id: ', counting_each_categoryid)
print('=> Total category ids: ', sum(counting_each_categoryid))

print('--- Val ---')
stat_id = []
stat_imageid = []
stat_categoryid = []
stat_len_seg = []
stat_iscrowd = []
stat_bbox = []
for idx, annot in enumerate(data_val['annotations']):
  stat_id.append(annot['id'])
  stat_imageid.append(annot['image_id'])
  stat_categoryid.append(annot['category_id'])
  stat_len_seg.append(len(annot['segmentation']))
  stat_iscrowd.append(annot['iscrowd'])
  stat_bbox.append(len(annot['bbox']))
  if idx == 0:
    print('- first bbox: ', annot['bbox'])
    print('- first area: ', annot['area'])

print('Unique ids: ', len(list(set(stat_id))))
print('Unique image ids: ', len(list(set(stat_imageid))))
print('Unique category ids: ', set(stat_categoryid))
print('Unique lens of segmentation: ', set(stat_len_seg))
print('Unique iscrowd: ', set(stat_iscrowd))
print('Unique lens of bbox', set(stat_bbox))

counting_each_categoryid = [0]*4
for categoryid in stat_categoryid:
  counting_each_categoryid[categoryid] += 1
print('=> Count each category id: ', counting_each_categoryid)
print('=> Total category ids: ', sum(counting_each_categoryid))


print('===== Categories =====')
n_categos = len(data['categories'])
for i in range(n_categos):
  print(data['categories'][i])

# total_files = 0
# lst_of_uniq_src_paths = []
# images = data['images']
# for image in images:
#   # print(image['file_name'])
#   src_path = image['file_name'].rsplit('/', 1)[0]
#   lst_of_uniq_src_paths.append(src_path)
#   total_files += 1
# f.close()

# print('=> Unique src paths: ', set(lst_of_uniq_src_paths))
# print('=> total_files: ', total_files)