model_name_or_path = "google/mt5-base"

# label paths
train_label_paths = {
    "BUTID": "./data/BUTID/train.labels",
    "BSign22k": "./data/BSign22k/train.labels",
    'AUTSL': './data/AUTSL/train.labels'
}

dev_label_paths = {
    "BUTID": "./data/BUTID/dev.labels",
    "BSign22k": "./data/BSign22k/dev.labels",
    'AUTSL': './data/AUTSL/dev.labels'
}

test_label_paths = {
    "BUTID": "./data/BUTID/test.labels",
    "BSign22k": "./data/BSign22k/test.labels",
    'AUTSL': './data/AUTSL/test.labels'
}

# pose paths
pose_dirs = {
    "BUTID": "/media/ks0085/storage/ks0085/datasets/BUTID/pose/openpose",
    "BSign22k": "/media/ks0085/storage/ks0085/datasets/BSign22k/pose/openpose",
    "AUTSL": "/media/ks0085/storage/ks0085/datasets/AUTSL/pose/openpose"
}