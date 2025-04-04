import os
import cv2
import torch
import argparse
import numpy as np
from memory_profiler import profile
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")
import time



def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(
            targetVideo) == 0:  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
# parser.add_argument('--video', dest='video', type=str, default=None)
video = 'desert.mp4'
# parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
# parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
# parser.add_argument('--model', dest='modelDir', type=str, default='train_log',
#                     help='directory with trained model files')
# parser.add_argument('--fp16', dest='fp16', action='store_true',
#                     help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
# parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
# parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
# parser.add_argument('--skip', dest='skip', action='store_true',
#                     help='whether to remove static frames before processing')
# parser.add_argument('--fps', dest='fps', type=int, default=None)
# parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
# parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
ext = 'mp4'
# parser.add_argument('--exp', dest='exp', type=int, default=1)
args = parser.parse_args()
# assert (not args.video is None or not args.img is None)
# if args.skip:
#     print("skip flag is abandoned, please refer to issue #207.")
# if args.UHD and args.scale == 1.0:
#     args.scale = 0.5
# assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
# if not args.img is None:
#     args.png = True

device = torch.device("cuda")
torch.set_grad_enabled(False)
# if torch.cuda.is_available():
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
    # if (args.fp16):
    #     torch.set_default_tensor_type(torch.cuda.HalfTensor)


# from train_log.RIFE_HDv3 import Model
from train_log.RIFE_HDv3 import Model

model = Model()
model.load_model('train_log', -1)
print("Loaded v3.x HD model.")
model.eval()
model.device()



videoCapture = cv2.VideoCapture(video)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()

fpsNotAssigned = True
fps_2 = fps * (2 ** 1)

videogen = skvideo.io.vreader(video)
lastframe = next(videogen)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_path_wo_ext, ext = os.path.splitext(video)
# print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, ext, tot_frame, fps, fps_2))

# if args.png == False and fpsNotAssigned == True:
#     print("The audio will be merged after interpolation process")
# else:
#     print("Will not merge audio because using png or fps flag!")

h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
# if args.png:
#     if not os.path.exists('vid_out'):
#         os.mkdir('vid_out')
# else:
# if args.output is not None:
#     vid_out_name = args.output
# else:
vid_out_name = '{}_{}X_{}fps{}'.format(video_path_wo_ext, (2 ** 1), int(np.round(fps_2)), ext)
vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps_2, (w, h))


def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        # if user_args.png:
        #     cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
        #     cnt += 1
        # else:
        vid_out.write(item[:, :, ::-1])


def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            # if user_args.montage:
            #     frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)


def make_inference(I0, I1):
    global model
    middle = model.inference(I0, I1,1.0)  #midlle是返回的是merged[2]
    return [middle]
    # first_half = make_inference(I0, middle, n=n // 2)
    # second_half = make_inference(middle, I1, n=n // 2)
    # if n % 2:
    #     return [*first_half, middle, *second_half]
    # else:
    #     return [*first_half, *second_half]


def pad_image(img):
    # if (args.fp16):
    #     return F.pad(img, padding).half()
    # else:
    return F.pad(img, padding)


# if args.montage:
#     left = w // 4
#     w = w // 2
tmp = max(32, int(32 / 1.0))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
# if args.montage:
#     lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None  # save lastframe when processing static frame

Time_Start = time.time() # 开始插帧

# ort_session = model.initialize_session() #onnx初始化
while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get()  # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, 1.0)#进行插帧
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    if ssim < 0.2:  #表示几乎没有相似性
        output = []
        for i in range((2 ** 1) - 1):
            output.append(I0)
        '''
        output = []
        step = 1 / (2 ** 1)
        alpha = 0
        for i in range((2 ** 1) - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(I0, I1)  #进行插帧

    # if args.montage:
    #     write_buffer.put(np.concatenate((lastframe, lastframe), 1))
    #     for mid in output:
    #         mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
    #         write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    # else:
    write_buffer.put(lastframe)
    for mid in output:
        mid = (((mid[0]*255.).byte().cpu().numpy().transpose(1,2,0)))
        # mid_tensor =torch.from_numpy(mid[0])
        # mid = (((mid_tensor * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

Time_End = time.time()
print("插帧用时为:{}".format(Time_End-Time_Start))
# if args.montage:
#     write_buffer.put(np.concatenate((lastframe, lastframe), 1))
# else:
write_buffer.put(lastframe)
write_buffer.put(None)


while (not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
# if args.png == False and fpsNotAssigned == True and not args.video is None:
if fpsNotAssigned == True and not video is None:
    try:
        transferAudio(video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)