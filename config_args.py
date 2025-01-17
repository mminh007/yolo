import yaml
import argparse

def setup_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str)
    parser.add_argument("--model", type=str)

    parser.add_argument("--data", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--in_chans", type=int)
    parser.add_argument("--imgsz", type=int)

    parser.add_argument("--S", type=list)
    parser.add_argument("--iou-thresh", type=float)
    parser.add_argument("--nms-thresh", type=float)
    parser.add_argument("--root", type=str)
    parser.add_argument("--nc", type=int)
    parser.add_argument("--names", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--dtype", type=str)
    parser.add_argument("--devices", type=str)
    parser.add_argument("--optimizer", type=str)
    
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--num-workers", type=int)
	
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--logger", type=str)

    return parser


def update_config(args: argparse.Namespace):
	if not args.config_file:
		return args

	cfg_path = args.config_file + ".yaml" if not args.config_file.endswith(".yaml") else args.config_file

	with open(cfg_path, "r") as f:
		data = yaml.load(f, Loader=yaml.FullLoader)
    
	for key, value in data.items():
		if getattr(args, key) is None:
			setattr(args, key, value)

    # config_args = argparse.Namespace(**data)
    # args = parser.parse_args(namespace=config_args)
    
	return args