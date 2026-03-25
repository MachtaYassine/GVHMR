import torch
from hmr4d.network.hmr2 import load_hmr2, HMR2


from hmr4d.utils.video_io_utils import read_video_np
import cv2
import numpy as np
import imageio.v3 as iio

from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from tqdm import tqdm


def get_batch(input_path, bbx_xys, img_ds=0.5, img_dst_size=256, path_type="video"):
    if path_type == "image":
        imgs = cv2.imread(str(input_path))[..., ::-1]
        imgs = cv2.resize(imgs, (0, 0), fx=img_ds, fy=img_ds)
        imgs = imgs[None]
    elif path_type == "np":
        assert isinstance(input_path, np.ndarray)
        assert img_ds == 1.0  # this is safe
        imgs = input_path
    elif path_type == "video":
        # Stream frames via pyav (same decoder as read_video_np) to avoid OOM on long videos
        filter_seq = []
        if img_ds != 1.0:
            filter_seq.append(("scale", f"iw*{img_ds}:ih*{img_ds}"))
        imgs_list = []
        bbx_xys_ds_list = []
        for i, frame_rgb in enumerate(iio.imiter(input_path, plugin="pyav", filter_sequence=filter_seq)):
            if i >= len(bbx_xys):
                break
            # Blur to avoid aliasing artifacts
            ds_factor = (bbx_xys[i, 2] * img_ds / img_dst_size / 2.0).item()
            if ds_factor > 1.1:
                frame_rgb = cv2.GaussianBlur(frame_rgb, (5, 5), (ds_factor - 1) / 2)
            # Crop to img_dst_size x img_dst_size
            crop, bbx_ds = crop_and_resize(
                frame_rgb,
                bbx_xys[i, :2] * img_ds,
                bbx_xys[i, 2] * img_ds,
                img_dst_size,
                enlarge_ratio=1.0,
            )
            imgs_list.append(crop)
            bbx_xys_ds_list.append(bbx_ds)
        imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
        bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)
        imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256)
        return imgs, bbx_xys

    # Non-video path: image/np — process all at once (small data)
    gt_center = bbx_xys[:, :2]
    gt_bbx_size = bbx_xys[:, 2]

    # Blur image to avoid aliasing artifacts
    gt_bbx_size_ds = gt_bbx_size * img_ds
    ds_factors = ((gt_bbx_size_ds * 1.0) / img_dst_size / 2.0).numpy()
    imgs = np.stack(
        [
            cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v
            for v, d in zip(imgs, ds_factors)
        ]
    )

    imgs_list = []
    bbx_xys_ds_list = []
    for i in range(len(imgs)):
        img, bbx_xys_ds = crop_and_resize(
            imgs[i],
            gt_center[i] * img_ds,
            gt_bbx_size[i] * img_ds,
            img_dst_size,
            enlarge_ratio=1.0,
        )
        imgs_list.append(img)
        bbx_xys_ds_list.append(bbx_xys_ds)
    imgs = torch.from_numpy(np.stack(imgs_list))  # (F, 256, 256, 3), RGB
    bbx_xys = torch.from_numpy(np.stack(bbx_xys_ds_list)) / img_ds  # (F, 3)

    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)  # (F, 3, 256, 256)
    return imgs, bbx_xys


class Extractor:
    def __init__(self, tqdm_leave=True):
        self.extractor: HMR2 = load_hmr2().cuda().eval()
        self.tqdm_leave = tqdm_leave

    def _auto_batch_size(self, sample_input, target_util=0.70):
        """Probe GPU to find optimal batch size.

        Uses two probe sizes to estimate both fixed overhead and per-sample cost,
        which is important for ViT models where attention memory scales with batch size.
        """
        device = sample_input.device
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Probe with bs=1 to get fixed overhead
        with torch.no_grad():
            _ = self.extractor({"img": sample_input})
        peak1 = torch.cuda.max_memory_allocated(device)
        del _
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Probe with bs=4 to get better per-sample estimate
        test_bs = 4
        test_input = sample_input.expand(test_bs, -1, -1, -1)
        with torch.no_grad():
            _ = self.extractor({"img": test_input})
        peak4 = torch.cuda.max_memory_allocated(device)
        per_sample = (peak4 - peak1) / (test_bs - 1)
        del test_input, _
        torch.cuda.empty_cache()

        total = torch.cuda.get_device_properties(device).total_memory
        available = total * target_util - peak1
        optimal = max(1, min(128, int(available / max(per_sample, 1))))
        # Safety: ViT attention/MLP intermediates scale worse than probes suggest.
        if total < 12e9:
            optimal = max(1, optimal // 4)
        print(f"  [Auto BS] HMR2 Feature: {total/1e9:.1f}GB GPU, {per_sample/1e6:.0f}MB/sample, fixed={peak1/1e6:.0f}MB -> batch_size={optimal}")
        return optimal

    def extract_video_features(self, video_path, bbx_xys, img_ds=0.5):
        """
        img_ds makes the image smaller, which is useful for faster processing
        """
        # Get the batch
        if isinstance(video_path, str):
            imgs, bbx_xys = get_batch(video_path, bbx_xys, img_ds=img_ds)
        else:
            assert isinstance(video_path, torch.Tensor)
            imgs = video_path

        # Inference
        F, _, H, W = imgs.shape  # (F, 3, H, W)
        # Keep frames on CPU, move only each batch to GPU to avoid OOM on small GPUs
        batch_size = self._auto_batch_size(imgs[:1].cuda())
        features = []
        for j in tqdm(range(0, F, batch_size), desc="HMR2 Feature", leave=self.tqdm_leave):
            imgs_batch = imgs[j : j + batch_size].cuda()

            with torch.no_grad():
                feature = self.extractor({"img": imgs_batch})
                features.append(feature.detach().cpu())
            del imgs_batch
            torch.cuda.empty_cache()

        features = torch.cat(features, dim=0).clone()  # (F, 1024)
        return features
