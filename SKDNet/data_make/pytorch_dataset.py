import os, cv2, tqdm
import numpy as np
from torch.utils.data import Dataset
## For new data generation.
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import data_make.dataset_utils as tools


class pytorch_dataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data =data

        ## Restrict the number of training and validation examples 
        if mode == 'train':
            if len(self.data) > 15000:
                self.data = self.data[:15000]
        elif mode == 'val':
            if len(self.data) > 3000:
                self.data = self.data[:3000]

        print('mode : {} the number of examples : {}'.format(mode, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        im_src_patch, im_dst_patch,  homography_src_2_dst, homography_dst_2_src = self.data[idx]

        return im_src_patch[0], im_dst_patch[0], homography_src_2_dst[0], homography_dst_2_src[0]

class DatasetGeneration(object):
    def __init__(self, args):
        self.size_patches = args.patch_size
        self.data_optical = args.data_dir_optical
        self.data_sar = args.data_dir_sar
        self.num_training = args.num_training_data
        self.synth_dir = args.synth_dir
        a=self.synth_dir
        self.check_directory(a)

        self.max_angle = args.max_angle
        self.min_scaling = args.min_scale
        self.max_scaling = args.max_scale
        self.max_shearing = args.max_shearing
        self.training_data = []
        self.validation_data = []

        if not self._init_dataset_path():
            self.images_info_optical = self._load_data_names(self.data_optical)
            self.images_info_sar = self._load_data_names(self.data_sar)

            self._create_synthetic_pairs(is_val=False)
            self._create_synthetic_pairs(is_val=True)
        else:
            self._load_synthetic_pairs(is_val=False)
            self._load_synthetic_pairs(is_val=True)

        print("#DMTTZDJC of Training / validation : ", len(self.training_data), len(self.validation_data))

    def check_directory(self,a):
        if not os.path.exists(a):
            os.mkdir(a)

    def get_training_data(self):
        return self.training_data

    def get_validation_data(self):
        return self.validation_data

    def _init_dataset_path(self):
        self.save_path = os.path.join(self.synth_dir ,'train_dataset')
        self.save_val_path = os.path.join(self.synth_dir , 'val_dataset')
        is_dataset_exists = os.path.exists(self.save_path) and os.path.exists(self.save_val_path)
        return is_dataset_exists

    def _load_data_names(self, data_dir):
        assert os.path.isdir(data_dir), "Invalid directory: {}".format(data_dir)
        count = 0
        images_info = []
        for r, d, f in os.walk(data_dir):
            for file_name in f:
                if file_name.endswith(".JPEG") or file_name.endswith(".jpg") or file_name.endswith(".png"):
                    images_info.append(os.path.join(r, file_name))
                    count += 1
        chang=len(np.asarray(images_info))

        src_idx = np.linspace(0,(chang-1),chang)
        src_idx=src_idx.astype('int64')
        images_info = np.asarray(images_info)[src_idx]
        print("Total images in directory at \" {} \" is : {}".format(data_dir, len(images_info)))
        return images_info

    def _create_synthetic_pairs(self, is_val):
        print('Generating Synthetic pairs . . . ' )
        paths = self._make_dataset_dir(is_val)

        if is_val:
            size_patches = self.size_patches
            self.counter += 1
        else:
            size_patches = self.size_patches
            self.counter = 0

        counter_patches = 0

        iterate_optical = tqdm.tqdm(range(len(self.images_info_optical)), total=len(self.images_info_optical), desc="DMTTZDJC dataset generation")

        for path_image_idx in iterate_optical:
            name_image_path_optoical = self.images_info_optical[(self.counter+path_image_idx) % len(self.images_info_optical)]
            name_image_path_sar = self.images_info_sar[(self.counter + path_image_idx) % len(self.images_info_sar)]
            correct_patch = False

            for _ in range(10):

                src_c = cv2.imread(name_image_path_optoical)
                dst_c = cv2.imread(name_image_path_sar)
                hom, scale, angle, _ = tools.generate_composed_homography(path_image_idx,self.max_angle, self.min_scaling,self.max_scaling, self.max_shearing)

                src, dst = self.generate_pair(src_c, dst_c,scale, angle, size_patches)

                if self._is_correct_size(src, dst, size_patches):
                    continue
                correct_patch = True
                break
            if correct_patch:
                im_src_patch = src.reshape((1, src.shape[2], src.shape[0], src.shape[1]))
                im_dst_patch = dst.reshape((1, dst.shape[2], dst.shape[0], dst.shape[1]))

                homography = self._generate_homography(src, hom, size_patches)
                homography_dst_2_src = self._preprocess_homography(homography)
                homography_src_2_dst = self._preprocess_homography(np.linalg.inv(homography))

                data = [im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src]

                self._update_data(data, is_val)

                self._save_synthetic_pair(paths, data, name_image_path_optoical.split('\\')[-1])

            counter_patches += 1
            if is_val and counter_patches >= 100:
                break
            elif counter_patches >= self.num_training:
                break
        self.counter = counter_patches


    def _make_dataset_dir(self, is_val):

        save_path = self.save_val_path if is_val else self.save_path

        self.check_directory('datasets')
        self.check_directory(save_path)

        path_im_src_patch = os.path.join(save_path, 'im_src_patch')
        path_im_dst_patch = os.path.join(save_path, 'im_dst_patch')
        path_homography_src_2_dst = os.path.join(save_path, 'homography_src_2_dst')
        path_homography_dst_2_src = os.path.join(save_path, 'homography_dst_2_src')

        self.check_directory(path_im_src_patch)
        self.check_directory(path_im_dst_patch)
        self.check_directory(path_homography_src_2_dst)
        self.check_directory(path_homography_dst_2_src)


        return path_im_src_patch, path_im_dst_patch,  path_homography_src_2_dst, path_homography_dst_2_src

    def _generate_homography(self, src, hom, size_patches):
        ## For GT-homography generation in image space
        inv_h = np.linalg.inv(hom)
        inv_h = inv_h / inv_h[2, 2]

        window_point = [src.shape[0] / 2, src.shape[1] / 2]
        point_src = [window_point[0], window_point[1], 1.0]
        point_dst = inv_h.dot([point_src[1], point_src[0], 1.0])
        point_dst = [point_dst[1] / point_dst[2], point_dst[0] / point_dst[2]]

        h_src_translation = np.asanyarray([[1., 0., -(int(point_src[1]) - size_patches / 2)],
                                           [0., 1., -(int(point_src[0]) - size_patches / 2)],
                                           [0., 0., 1.]])
        h_dst_translation = np.asanyarray([[1., 0., int(point_dst[1] - size_patches / 2)],
                                           [0., 1., int(point_dst[0] - size_patches / 2)],
                                           [0., 0., 1.]])

        homography = np.dot(h_src_translation, np.dot(hom, h_dst_translation))
        return homography

    def generate_pair(self,src_c, dst_c,scale, angle, size_patches):

        H_resize, W_resize = round(src_c.shape[0] / scale), round(src_c.shape[1] / scale)

        image_src = transforms.ToTensor()(src_c)
        image_dst = transforms.ToTensor()(dst_c)

        rotated_image = TF.rotate(image_dst, angle)
        rotated_scaled_image = TF.resize(rotated_image, size=(H_resize, W_resize))
        rotated_scaled_image = TF.center_crop(rotated_scaled_image, size_patches)

        cut_image_src = TF.center_crop(image_src, size_patches)


        src1 = np.asarray(TF.to_pil_image(cut_image_src))
        dst1 = np.asarray(TF.to_pil_image(rotated_scaled_image))

        src=cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst1, cv2.COLOR_BGR2GRAY)

        return src.reshape(src.shape[0],src.shape[1],1), dst.reshape(dst.shape[0],dst.shape[1],1)

    def _is_correct_size(self, src, dst, size_patches):
        return src.shape[0] != size_patches or src.shape[1] != size_patches or \
        dst.shape[0] != size_patches or dst.shape[1] != size_patches


    def _is_enough_edge(self,src,dst):
        ## (pre-processing) If src/dst image does not have "enough edge", then continue. (Sobel edge filter)
        src_sobelx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        src_sobelx = abs(src_sobelx.reshape((src.shape[0], src.shape[1], 1)))
        src_sobelx = src_sobelx.astype(float) / src_sobelx.max()       # 归一化

        dst_sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
        dst_sobelx = abs(dst_sobelx.reshape((dst.shape[0], dst.shape[1], 1)))
        dst_sobelx = dst_sobelx.astype(float) / dst_sobelx.max()

        label_dst_patch = dst_sobelx
        label_src_patch = src_sobelx

        is_enough_edge = not (label_dst_patch.max() < 0.25 or label_src_patch.max() < 0.25)
        return is_enough_edge


    def _preprocess_homography(self, h):
        h = h.astype('float32')
        h = h.flatten()
        h = h / h[8]
        h = h[:8]
        h = h.reshape((1, h.shape[0]))
        return h

    def _update_data(self, data, is_val):
        if is_val:
            self.validation_data.append(data)
        else:
            self.training_data.append(data)

    def _save_synthetic_pair(self, paths, data, name_image):
        ## Save the patches by np format (For caching)
        path_im_src_patch, path_im_dst_patch,path_homography_src_2_dst, path_homography_dst_2_src = paths
        im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src =  data

        np.save(os.path.join(path_im_src_patch, name_image), im_src_patch)
        np.save(os.path.join(path_im_dst_patch, name_image), im_dst_patch)
        np.save(os.path.join(path_homography_src_2_dst, name_image), homography_src_2_dst)
        np.save(os.path.join(path_homography_dst_2_src, name_image), homography_dst_2_src)

    def _load_synthetic_pairs(self, is_val):
        print('Loading Synthetic pairs . . .')

        path_im_src_patch, path_im_dst_patch, path_homography_src_2_dst, path_homography_dst_2_src, \
                        path_scale_src_2_dst, path_angle_src_2_dst = self._make_dataset_dir(is_val)

        for name_image in tqdm.tqdm(os.listdir(path_im_src_patch), total=len(os.listdir(path_im_src_patch))):
            if name_image[-8:] != "JPEG.npy":
                continue
            ## Load the patches by np format (caching)
            im_src_patch = np.load(os.path.join(path_im_src_patch, name_image))
            im_dst_patch = np.load(os.path.join(path_im_dst_patch, name_image))
            homography_src_2_dst = np.load(os.path.join(path_homography_src_2_dst, name_image))
            homography_dst_2_src = np.load(os.path.join(path_homography_dst_2_src, name_image))
            scale_src_2_dst = np.load(os.path.join(path_scale_src_2_dst, name_image))
            angle_src_2_dst = np.load(os.path.join(path_angle_src_2_dst, name_image))

            data = [im_src_patch, im_dst_patch, homography_src_2_dst, homography_dst_2_src, scale_src_2_dst, angle_src_2_dst]
            self._update_data(data, is_val)


def visualize_synthetic_pair(src_c, dst_c, im_src_patch, im_dst_patch, homography, size_patches, angle, scale):
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from kornia.geometry.transform import warp_perspective

    def _reconstruct_images(dst_c, scale, angle, size_patches, im_dst_patch, homography):
        ## recon 1 : using sca/ori
        H_recon, W_recon = round(dst_c.shape[0]*scale), round(dst_c.shape[1] * scale)
        dst_src_recon = transforms.ToTensor()(dst_c)
        dst_src_recon = TF.resize(dst_src_recon, size=(H_recon, W_recon)) ## scale
        dst_src_recon = TF.rotate(dst_src_recon, -angle)  ## angle
        dst_src_recon = TF.center_crop(dst_src_recon, size_patches)
        dst_src_recon = np.asarray(TF.to_pil_image(dst_src_recon))

        ## recon 2 : using homography
        dst_src_recon1 = warp_perspective(torch.tensor(im_dst_patch).to(torch.float32), torch.tensor(homography.astype('float32')).unsqueeze(0),  dsize=(size_patches, size_patches)).squeeze(0)
        dst_src_recon1 = np.asarray(TF.to_pil_image(dst_src_recon1.to(torch.uint8)))

        return dst_src_recon, dst_src_recon1

    dst_src_recon, dst_src_recon1 = _reconstruct_images(dst_c, scale, angle, size_patches, im_dst_patch, homography)
    src_dst_recon, src_dst_recon1 = _reconstruct_images(src_c, 1/scale, -angle, size_patches, im_src_patch, np.linalg.inv(homography))

    fig = plt.figure(figsize=(15,8))
    rows = 2 ; cols = 5
    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(src_c)
    ax1.set_title('src_c (input image)')
    ax1.axis("off")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(im_src_patch[0,0,:,:],  cmap='gray')
    ax2.set_title('im_src_patch')
    ax2.axis("off")

    ax3 = fig.add_subplot(rows, cols, 7)
    ax3.imshow(im_dst_patch[0,0,:,:],  cmap='gray')
    ax3.set_title('im_dst_patch')
    ax3.axis("off")

    ax4 = fig.add_subplot(rows, cols, 4)
