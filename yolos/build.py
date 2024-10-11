import yaml
import os
from utils.dataset import YoloDataset, ANCHORS
from torch.utils.data import DataLoader



ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


def create_data_config(path, num_classes, names):
    info = {
        "train": os.path.join(path, "train"),
        "val": os.path.join(path, "valid"),
        "test": os.path.join(path, "test"),
        "nc": num_classes,
        "names": names,
    }
    yaml_filepath = os.path.join(path, "data.yaml")
    with open(yaml_filepath, "w") as f:
        doc = yaml.dump(
            info,
            f,
            default_flow_style=None,
            sort_keys=False
        )


def get_loader(args, is_train=True):
    """
    """
    if args.data_file is None:
        data_file = os.join.path(args.root, "data.yaml")
        create_data_config(data_file, args.nc, names=None)

        with open(data_file, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    
    else:
        with open(args.data_file, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader) 
       

    dataset = YoloDataset(root_dir=data["train"] if is_train else data["val"],
                                anchors=ANCHORS,
                                imgsz=args.imgsz,
                                S=args.S,
                                C=args.nc,
                                transfroms = None)
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch,
                            num_workers=args.num_workers,
                            shuffle=True if is_train else False)
    
    return dataloader 