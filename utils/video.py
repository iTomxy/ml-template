import os, os.path as osp, shutil
import numpy as np
import cv2
from PIL import Image


def i2v(
    sorted_frame_files, # list of str, or list of list of str, full paths to sorted frame files
    dest_video_file,    # str, full path to save the video
    fps=24,
    fourcc="mp4v",
    overwrite=False,    # whether overwrite if `dest_video_file` already exists
    vstack=False,       # whether stack multiple videos vertically
):
    """concatenate frames into a video
    sorted_frame_files: List[str] or List[List[str]]
        If latter, will concatenate the frames within inner list.
    vstack: bool. If `sorted_frame_files` is List[List[str]],
        there are multiple videos to be concatenated. By default they
        will be concatenated horizontally, i.e. `hstack`. If set to true,
        they will be concatenate vetically.
    """
    assert overwrite or not osp.isfile(dest_video_file), dest_video_file
    os.makedirs(osp.dirname(dest_video_file) or '.', exist_ok=True)
    writer = None
    for fs in sorted_frame_files:
        if isinstance(fs, str): fs = [fs]
        img_ls, h, w = [], 0, 0
        for f in fs:
            assert osp.isfile(f), f
            img = cv2.imread(f)
            assert img is not None, f
            img_ls.append(img)
            _h, _w = img.shape[:2]
            if vstack:
                h, w = h + _h, max(w, _w)
            else:
                h, w = max(h, _h), w + _w
        if writer is None:
            _fourcc = cv2.VideoWriter_fourcc(*fourcc)
            writer = cv2.VideoWriter(dest_video_file, _fourcc, fps, (w, h), True)
        cat_frame, offset = np.zeros((h, w, 3), dtype=np.uint8), 0
        for img in img_ls:
            if vstack:
                cat_frame[offset: offset + img.shape[0], :img.shape[1]] = img
                offset += img.shape[0]
            else:
                cat_frame[:img.shape[0], offset: offset + img.shape[1]] = img
                offset += img.shape[1]
        writer.write(cat_frame)
    writer.release()


def v2i(
    src_video_file,     # str, full path to the video
    dest_dir,           # str, full path to folder to save the extracted frames
    save_type="png",    # extension to specify saving type, in {png, jpg, jpeg}
    overwrite=False,    # whether overwrite if `dest_dir` already exists
    head=0,             # if >0, only extract the first `head` frames
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
            if 0 < head <= i_frame:
                break

        if osp.isdir(dest_dir): shutil.rmtree(dest_dir)
        os.rename(tmp_dest_dir, dest_dir)
    finally:
        cap.release()


def i2gif(sorted_images, output_gif_file, duration=200, loop=0):
    """convert a image sequence into a gif
    Input:
        sorted_images: list of images, can be str, numpy.ndarray or PIL.Image.Image
        output_gif: str
        duration: int = 200, duration of each frame, in millisecond (ms)
        loop: int = 0, how many loops after initial play, 0 means infinite
    """
    images = []
    for im in sorted_images:
        if isinstance(im, Image.Image):
            images.append(im)
        elif isinstance(im, np.ndarray):
            images.append(Image.fromarray(im))
        elif isinstance(im, str):
            images.append(Image.open(str))

    os.makedirs(os.path.basename(output_gif) or '.', exist_ok=True)
    images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=loop)


if "__main__" == __name__:
    P = osp.join(os.environ["USERPROFILE"], "Videos")
    vf = osp.join(P, "test.mpvd")
    frame_dir = osp.join(P, "test-frames")
    v2i(vf, frame_dir, head=100)

    # detect edge of frames
    edge_dir = osp.join(P, "test-edges")
    os.makedirs(edge_dir, exist_ok=True)
    for f in os.listdir(frame_dir):
        img = cv2.imread(osp.join(frame_dir, f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(img, 20, 100)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(osp.join(edge_dir, osp.basename(f)), edge)

    # make a video of edge frame
    edge_frames = sorted(os.listdir(edge_dir), key=lambda f: int(f[:-4]))
    edge_frames = [osp.join(edge_dir, f) for f in edge_frames]
    i2v(edge_frames, osp.join(P, "test-edge.mp4"))
