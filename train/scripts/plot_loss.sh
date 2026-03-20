python tools/analysis_tools/analyze_logs.py plot_curve \
/project/hnguyen2/hqvo3/final_results/digital_pathology/MILxseg/run_6_classes_ps2048_loadfrom_3classes/20230531_002034_tmp.log.json \
--keys loss_cls loss_bbox loss_mask \
--legend loss_cls loss_bbox loss_mask \
--out ./vis/losses2.pdf