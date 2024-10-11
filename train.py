from yolos.build import get_loader, ANCHORS
from yolos.model.v3 import yolov3
from yolos.utils.process import nms
from yolos.utils.loss import YoloLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import logging
import os
import time

logger = logging.getLogger(__name__)

def main(args):
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    log_name = "log" + f"_yolov3_{args.data}.log"
    logging.basicConfig(filename=log_name, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",)

    
    logger.info(f"Create model: yolov3_{args.data}")

    model = yolov3(n_classes= args.num_classes)
    model.to(args.device)

    train_loader = get_loader(args)

    if args.optimizer is not None:
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[args.beta1, args.beta2],
                                   weight_decay=args.weight_decay)
            
        elif args.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=[args.beta1, args.beta2],
                                    weight_decay=args.weight_decay)

    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[args.beta1, args.beta2],
                                   weight_decay=args.weight_decay)

    criterion = YoloLoss()

    scaled_anchors = (
        torch.Tensor(ANCHORS) * torch.Tensor(args.S).unsqueeze(1).unsqueeze(1).repeate(1,3,2)
    ).to(args.device)

    train_loader = get_loader(args, is_train=True)
    val_loader = get_loader(args, is_train=False)

    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_loader)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch}")
    #start = time.time()

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
        epoch_losses = []
        for idx, (x, y) in loop:
               
            x = x.to(args.device)
            y0, y1, y2 = (
                y[0].to(args.device),
                y[1].to(args.device),
                y[2].to(args.device)
            )

            out = model(x)

            loss = (
                criterion(out[0], y0, scaled_anchors[0])
                + criterion(out[1], y1, scaled_anchors[1])
                + criterion(out[2], y2, scaled_anchors[2])
            )

            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        loop.set_description(f"Epoch [{epoch} / {args.epochs}] - Training")
        loop.set_postfix(loss=mean_loss)
        logging.info(f"Epoch {epoch+1}/{args.epochs} - Training Loss: {mean_loss:.4f}%")

        # evaluation
        model.eval()
        val_loss = []
        vloop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        with torch.no_grad():
            for idx, (x, y) in vloop:
                x = x.to(args.device)
                y0, y1, y2 = (
                    y[0].to(args.device),
                    y[1].to(args.device),
                    y[2].to(args.device)
                )

                out = model(x)

                loss = (
                    criterion(out[0], y0, scaled_anchors[0])
                    + criterion(out[1], y1, scaled_anchors[1])
                    + criterion(out[2], y2, scaled_anchors[2])
                )
                val_loss.append(loss.item())

        vloop.set_postfix(loss = sum(val_loss) / len(val_loss))
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Validation Loss: {sum(val_loss) / len(val_loss):.4f}%")
      
        # Save checkpoint
        model_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model checkpoint saved at {model_path}")
    
    print("Training completed")




            