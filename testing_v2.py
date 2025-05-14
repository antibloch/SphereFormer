import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import open3d as o3d
from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn_limit, collation_fn_voxelmean, collation_fn_voxelmean_tta
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from util.nuscenes import nuScenes
from util.semantic_kitti_v3 import SemanticKITTI_Pathetic
from util.waymo import Waymo

from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_stratified_transformer.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--test_path', type=str, default='test_data', help='test data')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # import torch.backends.mkldnn
    # ackends.mkldnn.enabled = False
    # os.environ["LRU_CACHE_CAPACITY"] = "1"
    # cudnn.deterministic = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # get model
    if args.arch == 'unet_spherical_transformer':
        from model.unet_spherical_transformer import Semantic as Model
        
        args.patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(input_c=args.input_c, 
            m=args.m,
            classes=args.classes, 
            block_reps=args.block_reps, 
            block_residual=args.block_residual, 
            layers=args.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / args.quant_size_scale, 
            quant_size_sphere=window_size_sphere / args.quant_size_scale, 
            rel_query=args.rel_query, 
            rel_key=args.rel_key, 
            rel_value=args.rel_value, 
            drop_path_rate=args.drop_path_rate, 
            window_size_scale=args.window_size_scale, 
            grad_checkpoint_layers=args.grad_checkpoint_layers, 
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    # set optimizer
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" not in n and p.requires_grad],
            "lr": args.base_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "transformer_block" in n and p.requires_grad],
            "lr": args.base_lr * args.transformer_lr_scale,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.base_lr, weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        writer = SummaryWriter(args.save_path)
        logger.info(args)

    if args.distributed:
        print(f"gpu: {gpu}")
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
        if args.get("max_grad_norm", None):
            logger.info("args.max_grad_norm = {}".format(args.max_grad_norm))

    # set loss func 
    class_weight = args.get("class_weight", None)
    class_weight = torch.tensor(class_weight).cuda() if class_weight is not None else None
    if main_process():
        logger.info("class_weight: {}".format(class_weight))
        logger.info("loss_name: {}".format(args.get("loss_name", "ce_loss")))
    criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=args.ignore_label, reduction='none' if args.loss_name == 'focal_loss' else 'mean').cuda()
    
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler_state_dict = checkpoint['scheduler']
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))



    val_transform = None
    args.use_tta = getattr(args, "use_tta", False)

    val_data = SemanticKITTI_Pathetic(data_path=args.data_root, 
        voxel_size=args.voxel_size,  
        rotate_aug=args.use_tta, 
        flip_aug=args.use_tta, 
        scale_aug=args.use_tta, 
        transform_aug=args.use_tta, 
        xyz_norm=args.xyz_norm, 
        pc_range=args.get("pc_range", None), 
        use_tta=args.use_tta,
        vote_num=args.vote_num,
        test_path='refined_data/pc',
        test_path_label='redined_data/label',
    )

    for i in range(len(val_data)):
        print(len(val_data))
        # sdfk

    gay_val_loader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=1, 
            shuffle=False, 
            # collate_fn=collation_fn_voxelmean
        )
    

    validate_distance(gay_val_loader, model, criterion)
    exit()



def focal_loss(output, target, class_weight, ignore_label, gamma, need_softmax=True, eps=1e-8):
    mask = (target != ignore_label)
    output_valid = output[mask]
    if need_softmax:
        output_valid = F.softmax(output_valid, -1)
    target_valid = target[mask]
    p_t = output_valid[torch.arange(output_valid.shape[0], device=target_valid.device), target_valid] #[N, ]
    class_weight_per_sample = class_weight[target_valid]
    focal_weight_per_sample = (1.0 - p_t) ** gamma
    loss = -(class_weight_per_sample * focal_weight_per_sample * torch.log(p_t + eps)).sum() / (class_weight_per_sample.sum() + eps)
    return loss


def validate_tta(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    for i, batch_data_list in enumerate(val_loader):

        data_time.update(time.time() - end)
    
        with torch.no_grad():
            output = 0.0
            for batch_data in batch_data_list:

                (coord, xyz, feat, target, offset, inds_reconstruct) = batch_data
                inds_reconstruct = inds_reconstruct.cuda(non_blocking=True)

                offset_ = offset.clone()
                offset_[1:] = offset_[1:] - offset_[:-1]
                batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

                coord = torch.cat([batch.unsqueeze(-1), coord], -1)
                spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
            
                coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
                batch = batch.cuda(non_blocking=True)

                sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size)

                assert batch.shape[0] == feat.shape[0]
                
                output_i = model(sinput, xyz, batch)
                output_i = F.softmax(output_i[inds_reconstruct, :], -1)
                
                output = output + output_i
            output = output / len(batch_data_list)
            
            if loss_name == 'focal_loss':
                loss = focal_loss(output, target, criterion.weight, args.ignore_label, args.loss_gamma)
            elif loss_name == 'ce_loss':
                loss = criterion(output, target)
            else:
                raise ValueError("such loss {} not implemented".format(loss_name))

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, mIoU, mAcc, allAcc



colors = {
    0: [0, 0, 0],       # unlabeled - black
    1: [0, 0, 1],       # car - blue
    2: [1, 0, 0],       # bicycle - red
    3: [1, 0, 1],       # motorcycle - magenta
    4: [0, 1, 1],       # truck - cyan
    5: [0.5, 0.5, 0],   # other-vehicle - olive
    6: [1, 0.5, 0],     # person - orange
    7: [1, 1, 0],       # bicyclist - yellow
    8: [1, 0, 0.5],     # motorcyclist - pink
    9: [0.5, 0.5, 0.5], # road - gray
    10: [0.5, 0, 0],    # parking - dark red
    11: [0, 0.5, 0],    # sidewalk - dark green
    12: [0, 0, 0.5],    # other-ground - dark blue
    13: [0, 0.5, 0.5],  # building - teal
    14: [0.5, 0, 0.5],  # fence - purple
    15: [0, 1, 0],      # vegetation - green
    16: [0.7, 0.7, 0.7],# trunk - light gray
    17: [0.7, 0, 0.7],  # terrain - light purple
    18: [0, 0.7, 0.7],  # pole - light cyan
    19: [0.7, 0.7, 0]   # traffic-sign - light yellow
}





def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create a video from the saved frames using OpenCV.
    
    Args:
        frame_dir (str): Directory containing the frames
        output_path (str): Path to save the video
        fps (int): Frames per second for the output video
    """
    # Get all PNG files in the directory
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        # Check if frame was loaded properly
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue
            
        video.write(frame)
    
    # Release the video writer
    video.release()
    print(f"Video saved to {output_path}")




def view_frame(points, pred_colors, labels_colors, legend_patches, output_dir, i):
    # Calculate center for rotation
    center = np.mean(points, axis=0)
    
    # Find the bounding box to determine appropriate axis limits
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    extent = np.max(max_bound - min_bound)
    
    # Fixed axis limits for consistent view
    view_radius = extent / 1.5  # Adjust for good default view

    p = 0.1
    max_zoom = 6.0

    # Calculate the rotation angle for this frame
    angle = p * 360.0
    
    # Create a rotation matrix around the y-axis
    rotation_matrix = np.array([
        [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
        [0, 1, 0],
        [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
    ])
    
    # Apply a constant zoom factor to all frames
    current_zoom = max_zoom
    
    # Apply rotation to the points relative to the center
    centered_points = points - center
    
    # Apply zoom by scaling the points (zoom in = points appear larger)
    zoomed_points = centered_points * current_zoom
    
    # Apply rotation and shift back
    rotated_points = np.dot(zoomed_points, rotation_matrix.T) + center

    # Create figure with two horizontal subplots
    fig = plt.figure(figsize=(20, 8), dpi=120)
    
    # First subplot - Predictions
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(
        rotated_points[:, 0], 
        rotated_points[:, 1], 
        rotated_points[:, 2],
        c=pred_colors, 
        s=0.5,
        alpha=0.7
    )
    ax1.set_title('Predictions', color='white', fontsize=14)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(center[0] - view_radius, center[0] + view_radius)
    ax1.set_ylim(center[1] - view_radius, center[1] + view_radius)
    ax1.set_zlim(center[2] - view_radius, center[2] + view_radius)
    ax1.set_axis_off()
    ax1.set_facecolor((0.1, 0.1, 0.1))
    elev = 20 + 10 * np.sin(np.radians(angle))
    ax1.view_init(elev=elev, azim=angle)
    
    # Second subplot - Ground Truth Labels
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(
        rotated_points[:, 0], 
        rotated_points[:, 1], 
        rotated_points[:, 2],
        c=labels_colors, 
        s=0.5,
        alpha=0.7
    )
    ax2.set_title('Ground Truth', color='white', fontsize=14)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(center[0] - view_radius, center[0] + view_radius)
    ax2.set_ylim(center[1] - view_radius, center[1] + view_radius)
    ax2.set_zlim(center[2] - view_radius, center[2] + view_radius)
    ax2.set_axis_off()
    ax2.set_facecolor((0.1, 0.1, 0.1))
    ax2.view_init(elev=elev, azim=angle)
    
    # Add legend to the second subplot
    ax2.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1),
              fancybox=True, shadow=True, ncol=1, fontsize='small',
              framealpha=0.8, facecolor='lightgray', edgecolor='black')
    
    # Set dark background for entire figure
    fig.set_facecolor((0.1, 0.1, 0.1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the image
    output_path = os.path.join(output_dir, f"frame_{i}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor=(0.1, 0.1, 0.1))
    plt.close()  # Close the figure to free memory





def validate_distance(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    class_names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
    "unlabeled",
    ]


    if os.path.exists('seg_results'):
        shutil.rmtree('seg_results')
    os.makedirs('seg_results', exist_ok=True)

    if os.path.exists('all_results'):
        shutil.rmtree('all_results')

    os.makedirs('all_results', exist_ok=True)

    # For validation on points with different distance
    intersection_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    union_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]
    target_meter_list = [AverageMeter(), AverageMeter(), AverageMeter()]

    torch.cuda.empty_cache()

    loss_name = args.loss_name

    model.eval()
    end = time.time()
    num_classes = 19
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)
    for i, batch_data in enumerate(val_loader):

        data_time.update(time.time() - end)



        coords, xyz, feats, labels, inds_recons, ref_points, fil_name = batch_data
        file_name_cloud = fil_name[0]
        coords = coords.squeeze(0)
        xyz = xyz.squeeze(0)
        feats = feats.squeeze(0)
        labels = labels.squeeze(0)
        inds_recons = inds_recons.squeeze(0)
        ref_points = ref_points.squeeze(0)
        ref_labels = labels.clone()

        print(coords.shape)
        print(xyz.shape)
        print(ref_points.shape)
        print("||||||||||||||||||||||")


        coords = (coords,)
        xyz = (xyz,)
        feats = (feats,)
        labels = (labels,)
        inds_recons = (inds_recons,)

        print(coords)
        print(xyz)
        print(feats)
        print(labels)
        print(inds_recons)


        inds_recons = list(inds_recons)

        accmulate_points_num = 0
        offset = []
        for i in range(len(coords)):
            inds_recons[i] = accmulate_points_num + inds_recons[i]
            accmulate_points_num += coords[i].shape[0]
            offset.append(accmulate_points_num)

        coord = torch.cat(coords)
        xyz = torch.cat(xyz)
        feat = torch.cat(feats)
        target = torch.cat(labels)
        offset = torch.IntTensor(offset)
        inds_reverse = torch.cat(inds_recons)



        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, args.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]
        
        output = output.max(1)[1]


        points_np = ref_points.cpu().numpy().astype(np.float32)
        segmentation_np = output.cpu().numpy()
        labels_np = labels.cpu().numpy().squeeze(-1)    

        print('points_np', points_np.shape)
        print('segmentation_np', segmentation_np.shape)
        print('labels_np', labels_np.shape)



        pred_colors = np.zeros((len(segmentation_np), 3), dtype=np.float32)
        ref_colors = np.zeros((len(labels_np), 3), dtype=np.float32)


        # computer per class IoU and mIOU


        for label in range(num_classes):
            pred_colors[segmentation_np == label] = colors[label]


        for label in range(num_classes+1):
            ref_colors[labels_np == label] = colors[label]

        legend_patches = []
        for i, class_name in enumerate(class_names):
            rgb_color = colors[i]
            patch = mpatches.Patch(color=rgb_color, label=class_name)   
            legend_patches.append(patch)


        view_frame(points_np,  pred_colors, ref_colors,  legend_patches, "all_results", i)




    

        valid_mask = (labels_np >= 0) & (labels_np < num_classes)
        valid_preds = segmentation_np[valid_mask]
        valid_labels = labels_np [valid_mask]
        for t, p in zip(valid_labels, valid_preds):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                conf_matrix[t, p] += 1



        colors = np.zeros((len(points_np), 3)).astype(np.float32)
        unique_labels = np.unique(segmentation_np)
        for i in unique_labels:
            mask = (segmentation_np == i)
            colors[mask] = np.array([np.random.rand(), np.random.rand(), np.random.rand()])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors)


        # save pcd
        o3d.io.write_point_cloud(os.path.join('all_results',f'{file_name_cloud}.pcd'), pcd)


    create_video_from_frames("all_results", os.path.join("all_results", "output_video.mp4"), fps=2)

   

    print("\n==== Per-Class IoU and mIoU ====")
    ious = []
    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        denom = TP + FP + FN
        if denom == 0:
            iou = float('nan')
        else:
            iou = TP / denom
        ious.append(iou)
        print(f"{class_names[i]:<15}: IoU = {iou:.4f}" if not np.isnan(iou) else f"{class_names[i]:<15}: IoU = N/A (class absent)")

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious)
    print(f"\nMean IoU over {len(valid_ious)} valid classes: {mean_iou:.4f}")
    
 

if __name__ == '__main__':
    import gc
    gc.collect()
    main()
