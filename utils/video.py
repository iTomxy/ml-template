import os, os.path as osp, shutil
import cv2


def i2v(
    sorted_frame_files, # list of sorted frame files full path
    dest_video_file,    # str, full path to save the video
    fps=24,
    fourcc="mp4v",
    overwrite=False     # whether overwrite if `dest_video_file` already exists
):
    """concatenate frames into a video"""
    assert overwrite or not osp.isfile(dest_video_file), dest_video_file
    os.makedirs(osp.dirname(dest_video_file) or '.', exist_ok=True)
    writer = None
    for f in sorted_frame_files:
        assert osp.isfile(f), f
        img = cv2.imread(f)
        assert img is not None, f
        if writer is None:
            h, w = img.shape[:2]
            _fourcc = cv2.VideoWriter_fourcc(*fourcc)
            writer = cv2.VideoWriter(dest_video_file, _fourcc, FPS, (w, h), True)
        writer.write(img)
    writer.release()


def v2i(
    src_video_file,     # str, full path to the video
    dest_dir,           # str, full path to folder to save the extracted frames
    save_type="png",    # extension to specify saving type, in {png, jpg, jpeg}
    overwrite=False,    # whether overwrite if `dest_dir` already exists
):
    """extract video frames"""
    assert osp.isfile(src_video_file), src_video_file
    assert overwrite or not osp.isdir(dest_dir), dest_dir
    assert save_type.lower() in ("png", "jpg", "jpeg"), save_type
    cap = cv2.VideoCapture(src_video_file)
    assert cap.isOpened(), "* Open failed: " + src_video_file
    try:
        # trim trailing path separator
        while dest_dir[-1] in "\\/": dest_dir = dest_dir[:-1]
        tmp_dest_dir = dest_dir + ".tmp"
        os.makedirs(tmp_dest_dir, exist_ok=True)

        i_frame = 0
        while True:
            flag, frame = cap.read()
            if not flag: break
            cv2.imwrite(osp.join(tmp_dest_dir, str(i_frame)+'.'+save_type), frame)
            i_frame += 1

        if osp.isdir(dest_dir): shutil.rmtree(dest_dir)
        os.rename(tmp_dest_dir, dest_dir)
    finally:
        cap.release()

