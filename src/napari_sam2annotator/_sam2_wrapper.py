import napari
import skimage as ski
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import glob
from sam2.build_sam import build_sam2_video_predictor
from ._funcs import delete_tmp_files, convert_to_rgb, cleanup_mask, is_legit_shape


class sam2_wrapper:
    def __init__(self, model_path='U:/smc/public/SMC/sam2/checkpoints/sam2_hiera_base_plus.pt', updated=True):
        """
        Initialize the SAM2 wrapper.

        Args:
            model_path (str): Path to the SAM2 model.
            device (str): Device to use for inference. Default is "cuda".
        """
        sam2_checkpoint = model_path
        #model_cfg = "sam2_hiera_l.yaml"
        model_cfg = "sam2_hiera_b+.yaml"

        device = torch.device("cuda")
        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        if updated:
            sam2_checkpoint = 'U:/smc/public/SMC/sam2.1/sam2/checkpoints/sam2.1_hiera_large.pt'
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


    def set_image(self, img, channel=0, channel_axis=-10):
        if channel_axis > -10:
            img = np.moveaxis(img, channel_axis, 0)
        if channel_axis>-10:
            self.img = img[channel]
        else:
            self.img = img
        delete_tmp_files()
        convert_to_rgb(self.img)
        self.inference_state = self.predictor.init_state(video_path='tmp/')

    def infer_from_box(self, shapes, max_distance=90000, do_reset=True):

        if do_reset:
            self.predictor.reset_state(self.inference_state)
        self.number_objects = len(shapes)
        # Add these shapes to the predictor
        for idx, shape in enumerate(shapes):
            rect = shape
            if not is_legit_shape(rect):
                print('Bad shape')
                continue
            box = np.array([np.min(rect[:,2]), np.min(rect[:,1]), np.max(rect[:,2]), np.max(rect[:,1])]).astype(int)
            self.ann_frame_idx = int(rect[0][0])
            ann_obj_id = idx+1

            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
                )

        full_mask = self.propagate_back_forth(max_distance=max_distance)
        
        return full_mask

    def infer_from_box_single_object(self, shapes, max_distance=90000, do_reset=True):

        if do_reset:
            self.predictor.reset_state(self.inference_state)
        self.number_objects = 1
        # Add these shapes to the predictor
        for idx, shape in enumerate(shapes):
            rect = shape
            if not is_legit_shape(rect):
                print('Bad shape')
                continue
            box = np.array([np.min(rect[:,2]), np.min(rect[:,1]), np.max(rect[:,2]), np.max(rect[:,1])]).astype(int)
            self.ann_frame_idx = int(rect[0][0])
            ann_obj_id = 1

            
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
            )

        full_mask = self.propagate_back_forth(max_distance=max_distance)
        return full_mask
    
    def infer_from_points(self, good_points, bad_points, do_reset=False):
        if do_reset:
            self.predictor.reset_state(self.inference_state)

        self.number_objects = 1

        for ann_frame_idx in range(self.img.shape[0]):
            current_good_points = good_points[good_points[:,0] == ann_frame_idx]
            current_bad_points = bad_points[bad_points[:,0] == ann_frame_idx]
            current_points = np.concatenate([current_good_points, current_bad_points], axis=0)
            if len(current_points) == 0:
                continue
            ann_obj_id = 1
            labels = len(current_good_points)*[1] + len(current_bad_points)*[0]

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=current_points[:,::-1][:,0:2].copy(),
                labels=labels,
            )
        full_mask = self.propagate_back_forth(max_distance=90000)
        return full_mask


    def propagate_back_forth(self, max_distance=90000):
        # Propagate all of them backwards and forwards

        # Forward
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        full_mask = np.zeros_like(self.img[:,:,:])
        for frame_idx, segmentations in video_segments.items():
            for obj in np.arange(1, self.number_objects+1):
                cur_mask = segmentations[obj][0]
                if np.abs(frame_idx-self.ann_frame_idx) < max_distance:
                    full_mask[frame_idx] = full_mask[frame_idx] + cur_mask*obj

        # Reverse
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        for frame_idx, segmentations in video_segments.items():
            for obj in np.arange(1, self.number_objects+1):
                # if frame_idx == int(self.rect[0][0]):
                #     continue
                if np.sum(full_mask[frame_idx]==obj) > 0:
                    continue
                cur_mask = segmentations[obj][0]
                if np.abs(frame_idx-self.ann_frame_idx) < max_distance:
                    full_mask[frame_idx] = full_mask[frame_idx] + cur_mask*obj

        # Make sure it does not jerk too much
        if self.number_objects==1:
            for lab in range(1,int(np.max(full_mask))):
                full_mask = cleanup_mask(full_mask, 1, self.ann_frame_idx)
        return full_mask