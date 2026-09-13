from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg)
        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history

    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def get_one_track(self, video_path, rank=0):
        """Bbox track for the person at `rank` in the area-sorted list (0 = largest).

        vid2smplx patch. Also records every track it saw in self.track_summary: the caller
        reconstructs ONE person, and a downstream consumer cannot tell that from "there was
        only one person" unless the discarded tracks are reported. Each entry carries the
        two numbers that separate a real second subject from tracker noise:
          n_overlap_frames -- frames this track shares with the chosen one. A YOLO id
            switch (the SAME person re-identified) overlaps ~0 frames; a second person
            standing in shot overlaps nearly all of them.
          area_share -- this track's median bbox area over the chosen track's. A distant
            passer-by or a reflection is a small fraction; a co-present subject is O(1).
        """
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        if len(id_sorted) == 0:
            raise RuntimeError(
                f"No person was detected anywhere in {video_path}. GVHMR needs a visible person "
                f"(YOLO confidence >= 0.5); check the clip and trim to a section where one is in frame."
            )
        if rank >= len(id_sorted):
            raise RuntimeError(
                f"--person {rank} was requested but only {len(id_sorted)} person track(s) were found "
                f"in {video_path}; valid ranks are 0..{len(id_sorted) - 1} (0 = largest in frame)."
            )
        track_id = id_sorted[rank]

        _L, _W, _H = get_video_lwh(video_path)

        def _areas(tid):
            _b = np.asarray(id_to_bbx_xyxys[tid], dtype=np.float64)
            _wh = _b[:, 2:] - _b[:, :2]
            return _wh[:, 0] * _wh[:, 1] / _W / _H

        _chosen_frames = set(id_to_frame_ids[track_id])
        _chosen_area = float(np.median(_areas(track_id)))
        self.track_summary = {
            "n_tracks": len(id_sorted),
            "n_frames_total": int(_L),
            "chosen_rank": rank,
            "chosen_track_id": int(track_id),
            "tracks": [
                {
                    "rank": i,
                    "track_id": int(tid),
                    "n_frames": len(id_to_frame_ids[tid]),
                    "n_overlap_frames": len(_chosen_frames & set(id_to_frame_ids[tid])),
                    "area_median": round(float(np.median(_areas(tid))), 6),
                    "area_share": (round(float(np.median(_areas(tid))) / _chosen_area, 4)
                                   if _chosen_area > 0 else 0.0),
                    "bbx_xyxy_median": np.median(id_to_bbx_xyxys[tid], axis=0).round(1).tolist(),
                }
                for i, tid in enumerate(id_sorted)
            ],
        }
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track
