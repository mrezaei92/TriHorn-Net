import torch
from utils.utils import *
import time
import tqdm
from utils.utils import model_builder
from dataloader import DATASET_NUM_JOINTS
from utils.forwardpass import get_forwardPass


def Train(model, data_loaders, args,lossFunction, optimizer, device, scheduler, fp16_scaler, rank):
    model.train()

    ForwardPassFunc = get_forwardPass(args.train_mode,args)
    
    trainloader_labeled = data_loaders["trainloader_labeled"]

    for epoch in range(args.num_epoch):

        if args.paralelization_type=="DDP":
            trainloader_labeled.sampler.set_epoch(epoch)

        meter = AverageMeter(fmt=':.3f')
        
        start_time_iter = time.time()
        start_time_iter2 = time.time()
        
        loop = tqdm.tqdm(trainloader_labeled)
        loop.set_description(f"Epoch {epoch+1}: ")
        
        for i, data in enumerate(loop, 0):
            
            if args.paralelization_type=="DDP":
                inputs, gt_uvd, com, cubesize , joint_mask, visible_mask = data[0].cuda(device, non_blocking=True), data[1].cuda(device, non_blocking=True), data[4].cuda(device, non_blocking=True), data[6].cuda(device, non_blocking=True), data[7].cuda(device, non_blocking=True), data[8].cuda(device, non_blocking=True)           
            else:
                inputs, gt_uvd, com, cubesize , joint_mask, visible_mask= data[0].to(device), data[1].to(device), data[4].to(device), data[6].to(device), data[7].to(device), data[8].to(device)
            
            scale=args.cubic_size/2.

            gt_uvd=Normalize_depth(gt_uvd,sizes=cubesize,coms=com,add_com=False).float()

            # forward + backward + optimize 
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                loss, loss_dict = ForwardPassFunc(model, inputs, gt_uvd, lossFunction, cubesize, com, joint_mask, visible_mask, args)
                
            
            meter.update(loss_dict)

            Grad_Updater(loss, model, optimizer, fp16_scaler, args)

           

        ## End of epoch
        meter.synchronize_between_processes()
        
        message=f"End of epoch: {epoch+1}: " + str(meter) + f" | Total Time: {(time.time()-start_time_iter2)/60:.2f} mins\n"
        print(message)

        if args.scheduler == "auto":
            scheduler.step( meter.averages["Tot_loss"] )
        else:
            scheduler.step()

        # Save the model
        if is_main_process() and ( (args.num_epoch-epoch)<=20 or (epoch+1)%args.save_freq==0 ) and epoch!=0:
            model_name="savedModel_E{}.pt".format(epoch+1)
            make_checkpoint(model_name, model, optimizer, scheduler, args)




    return
