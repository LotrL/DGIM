import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os
import re
import cv2
import random
import warnings
import numpy as np
import albumentations as A

from src.fast_neural_style import utils
from src.fast_neural_style.transfer_net import TransferNet

warnings.filterwarnings("ignore")


class MultiModality(Dataset):
    def __init__(self,
                 root_dir,
                 mode='train',
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 augment_fn=None,
                 **kwargs):
        """
        Manage one scene(npz_path) of Multi-Modality dataset.

        Args:
            root_dir (str): multi-modality root directory that has `phoenix`.
            mode (str): options are ['train', 'val', 'test'].
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = []
        self.build_dataset()

        # parameters for image resizing and padding
        # if mode == 'train':
        #     assert img_resize is not None and img_padding
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding

        # for training DGIM
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)

        # build style path list and style model
        self.style_folder_path = "/root/my_project/DoGFTR_v7/src/fast_neural_style/styles"
        self.style_path_list = self.build_style_path_list()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_model_path = "/root/my_project/DoGFTR_v7/src/fast_neural_style/weights/epoch5.pth"
        self.style_model = self.build_style_model()

    def build_dataset(self):
        # multi-modal dataset
        scenes = sorted(os.listdir(self.root_dir))
        for scene in scenes:
            scene_path = self.root_dir + f"/{scene}"
            scene_pairs = sorted(os.listdir(scene_path))
            for scene_pair in scene_pairs:
                scene_pair_path = scene_path + f"/{scene_pair}"
                folders = sorted(os.listdir(scene_pair_path))
                for folder in folders:
                    folder_path = scene_pair_path + f"/{folder}"
                    files = sorted(os.listdir(folder_path))
                    for i in range(3, len(files)):  # pass the first pair
                        if i % 2 == 1:
                            self.dataset.append({"image0_path": f"{folder_path}/{files[0]}",
                                                 "image1_path": f"{folder_path}/{files[i]}",
                                                 "affine_matrix_path": f"{folder_path}/{files[i + 1]}"})
        np.random.seed(42)
        np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # TODO: Support augmentation & handle seeds for each worker correctly.

        # ----- pretrain -----
        # stylizing_option0, stylizing_option1 = False, False

        # thermal_option0, thermal_option1 = False, False

        # ----- finetune -----
        if random.choice([0, 1]) == 0:
            stylizing_option0, stylizing_option1 = True, False
        else:
            stylizing_option0, stylizing_option1 = False, True

        if random.choice([0, 1]) == 0:
            thermal_option0, thermal_option1 = True, False
        else:
            thermal_option0, thermal_option1 = False, True

        # noinspection PyTypeChecker
        image0, mask0, scale0 = self.read_megadepth_gray(
            item["image0_path"], self.img_resize, self.df, self.img_padding, None, stylizing_option0, thermal_option0)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # noinspection PyTypeChecker
        image1, mask1, scale1 = self.read_megadepth_gray(
            item["image1_path"], self.img_resize, self.df, self.img_padding, None, stylizing_option1, thermal_option1)
        # np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read and compute relative poses
        # noinspection PyTypeChecker
        T_0to1 = torch.from_numpy(np.loadtxt(item["affine_matrix_path"]).astype(np.float32))  # (3, 3)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,  # (3, 3)
            'T_1to0': T_1to0,
            'scale0': scale0,  # [scale_w, scale_h]
            'scale1': scale1,
            'dataset_name': 'Multi-Modality',
            'pair_id': idx,
        }

        # for DGIM training
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                       scale_factor=self.coarse_scale,
                                                       mode='nearest',
                                                       recompute_scale_factor=False)[0].bool()
            data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data

    def build_style_path_list(self):
        style_path_list = []
        style_names = os.listdir(self.style_folder_path)
        for style_name in style_names:
            style_path = self.style_folder_path + "/" + style_name
            style_path_list.append(style_path)
        return style_path_list

    def build_style_model(self):
        style_model = TransferNet()
        state_dict = torch.load(self.style_model_path)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r"in\d+\.running_(mean|var)$", k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(self.device).eval()
        # Freeze model parameters
        for param in style_model.parameters():
            param.requires_grad = False
        return style_model

    def stylize(self, content_path, style_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        content_image = utils.load_image(content_path)
        content_image = transform(content_image)
        content_image = content_image.unsqueeze(0).to(self.device)

        style_image = utils.load_image(style_path)
        style_image = style_image.resize((256, 256))
        style_image = transform(style_image)
        style_image = style_image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.style_model(content_image, style_image)
            output = output.squeeze(0).clamp(0, 255).cpu().numpy().transpose(1, 2, 0).astype("uint8")
        torch.cuda.empty_cache()

        # thermal_transform = BGRtoThermal()
        # guide_image = thermal_transform.augment_pseudo_thermal(cv2.imread(content_path))  # thermal image
        guide_image = cv2.imread(content_path)  # original image

        output_image = cv2.ximgproc.guidedFilter(guide=guide_image,
                                                 src=cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
                                                 radius=15, eps=0.001)
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)  # stylized image
        # output_image = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)  # thermal or original image
        return output_image

    def imread_gray(self, path, stylizing_option, thermal_option):
        if stylizing_option:
            image = self.stylize(path, random.choice(self.style_path_list))
        else:
            image = cv2.imread(path)

        if thermal_option:
            thermal_transform = BGRtoThermal()
            image = thermal_transform.augment_pseudo_thermal(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image  # (h, w)

    def read_megadepth_gray(self, path, resize=None, df=None, padding=False, augment_fn=None, stylizing_option=False, thermal_option=False):
        """
        Args:
            resize (int, optional): the longer edge of resized images. None for no resize.
            padding (bool): If set to 'True', zero-pad resized images to squared size.
            augment_fn (callable, optional): augments images with pre-defined visual effects
        Returns:
            image (torch.tensor): (1, h, w)
            mask (torch.tensor): (h, w)
            scale (torch.tensor): [w/w_new, h/h_new]
        """
        # read image
        image = self.imread_gray(path, stylizing_option, thermal_option)

        # resize image
        w, h = image.shape[1], image.shape[0]
        w_new, h_new = self.get_resized_wh(w, h, resize)
        w_new, h_new = self.get_divisible_wh(w_new, h_new, df)

        image = cv2.resize(image, (w_new, h_new))
        scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

        if padding:  # padding
            pad_to = max(h_new, w_new)
            image, mask = self.pad_bottom_right(image, pad_to, ret_mask=True)
        else:
            mask = None

        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
        mask = torch.from_numpy(mask)

        return image, mask, scale

    def get_resized_wh(self, w, h, resize=None):
        if resize is not None:  # resize the longer edge
            scale = resize / max(h, w)
            w_new, h_new = int(round(w * scale)), int(round(h * scale))
        else:
            w_new, h_new = w, h
        return w_new, h_new

    def get_divisible_wh(self, w, h, df=None):
        if df is not None:
            w_new, h_new = map(lambda x: int(x // df * df), [w, h])
        else:
            w_new, h_new = w, h
        return w_new, h_new

    def pad_bottom_right(self, inp, pad_size, ret_mask=False):
        assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
        mask = None
        if inp.ndim == 2:
            padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
            padded[:inp.shape[0], :inp.shape[1]] = inp
            if ret_mask:
                mask = np.zeros((pad_size, pad_size), dtype=bool)
                mask[:inp.shape[0], :inp.shape[1]] = True
        elif inp.ndim == 3:
            padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
            padded[:, :inp.shape[1], :inp.shape[2]] = inp
            if ret_mask:
                mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
                mask[:, :inp.shape[1], :inp.shape[2]] = True
        else:
            raise NotImplementedError()
        return padded, mask


class BGRtoThermal:
    def __init__(self):
        self.blur = A.Blur(p=0.7, blur_limit=(3, 5))  # default: blur_limit=(2, 4)
        self.hsv = A.HueSaturationValue(p=0.9, val_shift_limit=(-30, +30), hue_shift_limit=(-90, +90),
                                        sat_shift_limit=(-30, +30))

        # parameters for the cosine transform
        self.w_0 = np.pi * 2 / 3
        self.w_r = np.pi / 2
        self.theta_r = np.pi / 2

    def augment_pseudo_thermal(self, image):
        # HSV augmentation
        image = self.hsv(image=image)["image"]

        # Random blur
        image = self.blur(image=image)["image"]

        # Convert the image to the gray scale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image between (-0.5, 0.5)
        image = image / 255 - 0.5

        # Random phase and freq for the cosine transform
        phase = np.pi / 2 + np.random.randn(1) * self.theta_r
        w = self.w_0 + np.abs(np.random.randn(1)) * self.w_r

        # Cosine transform
        image = np.cos(image * w + phase)

        # Min-max normalization for the transformed image
        image = (image - image.min()) / (image.max() - image.min()) * 255

        # 3 channel gray
        # image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return image.astype(np.uint8)
