from inference import convert_video
import torch
import glob
import os
import shutil

for movie in glob.glob("input/*.mp4"):
    basename = os.path.splitext(os.path.basename(movie))[0]
    basename_ex = os.path.basename(movie)
    if not os.path.exists(os.path.join("output", basename)):
        os.mkdir(os.path.join("output", basename))
    shutil.copy(movie, os.path.join("output", basename, basename_ex))
    convert_video(
        model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda(),
        input_source=movie,
        output_type='video',
        output_composition=os.path.join("output", basename, "com.mp4"),
        output_alpha=os.path.join("output", basename, "pha.mp4"),
        output_foreground=os.path.join("output", basename, "fgr.mp4"),
        output_video_mbps=4,
        downsample_ratio=None,
        seq_chunk=12,
    )
