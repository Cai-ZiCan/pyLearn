# Original Dataset Structure

## Building Change Detection Dataset (raw)

This dataset contains two periods (2012, 2016) with split tiles and whole images, plus change labels and shapefiles.

```
Building change detection dataset_add/
├─ 1. The two-period image data/
│  ├─ 2012/
│  │  ├─ splited_images/
│  │  │  ├─ test/
│  │  │  │  ├─ image/
│  │  │  │  └─ label/
│  │  │  └─ train/
│  │  │     ├─ image/
│  │  │     └─ label/
│  │  └─ whole_image/
│  │     ├─ test/
│  │     │  ├─ image/   (2012_test.tfw/.tif.aux.xml/.tif.ovr/.tif.xml)
│  │     │  └─ label/   (matching label files)
│  │     └─ train/
│  │        ├─ image/   (2012_train.*)
│  │        └─ label/   (matching label files)
│  ├─ 2016/
│  │  ├─ splited_images/
│  │  │  ├─ test/
│  │  │  │  ├─ image/
│  │  │  │  └─ label/
│  │  │  └─ train/
│  │  │     ├─ image/
│  │  │     └─ label/
│  │  └─ whole_image/
│  │     ├─ test/
│  │     │  ├─ image/   (2016_test.*)
│  │     │  └─ label/   (matching label files)
│  │     └─ train/
│  │        ├─ image/   (2016_train.*)
│  │        └─ label/   (matching label files)
│  └─ change_label/
│     ├─ test/   (change_label.*)
│     └─ train/  (change_label.*)
└─ 2. The shape file of the images/
	├─ test/   (test.shp/.shx/.dbf/.sbn/.sbx)
	└─ train/  (train.shp/.shx/.dbf/.sbn/.sbx)
```

# use for training Datafile

## WHU_build (current task)

This folder contains prepared training/testing splits, change detection labels, and prediction outputs.

```
WHU_build/
├─ changed_data/
│  ├─ img/
│  │  ├─ test/
│  │  │  ├─ 2012/
│  │  │  └─ 2016/
│  │  └─ train/
│  │     ├─ 2012/
│  │     └─ 2016/
│  └─ label/
│     ├─ test/
│     │  ├─ 2012/
│     │  └─ 2016/
│     └─ train/
│        ├─ 2012/
│        └─ 2016/
└─ split_data/
	├─ predict/
	│  ├─ 2016_train.tfw/.tif.aux.xml/.tif.ovr/.tif.xml
	│  └─ result/
	├─ test/
	│  ├─ image/
	│  └─ label/
	└─ train/
		├─ image/
		└─ label/
```
