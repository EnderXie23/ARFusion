import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse
from networks import *
from skyboxengine import *
import utils
import torch
import threading
import queue
import time

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='SKYAR')
parser.add_argument('--path', type=str, default='./config/config-canyon-jupiter.json', metavar='str',
                    help='configurations')

class SkyFilter():

    def __init__(self, args):

        self.ckptdir = args.ckptdir
        self.datadir = args.datadir
        self.input_mode = args.input_mode

        self.in_size_w, self.in_size_h = args.in_size_w, args.in_size_h
        self.out_size_w, self.out_size_h = args.out_size_w, args.out_size_h

        self.skyboxengine = SkyBox(args)

        self.net_G = define_G(input_nc=3, output_nc=1, ngf=64, netG=args.net_G).to(device)
        self.load_model()
        # self.net_G = torch.compile(self.net_G)

        self.video_writer = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (args.out_size_w, args.out_size_h))
        self.video_writer_cat = cv2.VideoWriter('demo-cat.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (2*args.out_size_w, args.out_size_h))

        if args.save_jpgs and os.path.exists(args.output_dir) is False:
            os.mkdir(args.output_dir)

        self.save_jpgs = args.save_jpgs


    def load_model(self):

        print('loading the best checkpoint...')
        checkpoint = torch.load(os.path.join(self.ckptdir, 'best_ckpt.pt'),
                                map_location=None if torch.cuda.is_available() else device)
        # checkpoint = torch.load(os.path.join(self.ckptdir, 'last_ckpt.pt'))
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()



    def write_video(self, img_HD, syneth):

        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        self.video_writer.write(frame)

        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)
        self.video_writer_cat.write(frame_cat)

        # cv2.imshow('frame_cat', frame_cat)
        # cv2.waitKey(1)



    def synthesize(self, img_HD, img_HD_prev):

        h, w, c = img_HD.shape

        img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))

        img = np.array(img, dtype=np.float32)
        img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)

        with torch.no_grad():
            G_pred = self.net_G(img.to(device))
            G_pred = torch.nn.functional.interpolate(G_pred, (h, w), mode='bicubic', align_corners=False)
            G_pred = G_pred[0, :].permute([1, 2, 0])
            G_pred = torch.cat([G_pred, G_pred, G_pred], dim=-1)
            G_pred = np.array(G_pred.detach().cpu())
            G_pred = np.clip(G_pred, a_max=1.0, a_min=0.0)

        skymask = self.skyboxengine.skymask_refinement(G_pred, img_HD)

        syneth = self.skyboxengine.skyblend(img_HD, img_HD_prev, skymask)

        return syneth, G_pred, skymask



    def cvtcolor_and_resize(self, img_HD):

        img_HD = cv2.cvtColor(img_HD, cv2.COLOR_BGR2RGB)
        img_HD = np.array(img_HD / 255., dtype=np.float32)
        img_HD = cv2.resize(img_HD, (self.out_size_w, self.out_size_h))

        return img_HD



    def run_imgseq(self):

        print('running evaluation...')
        img_names = os.listdir(self.datadir)
        img_HD_prev = None

        for idx in range(len(img_names)):

            this_dir = os.path.join(self.datadir, img_names[idx])
            img_HD = cv2.imread(this_dir, cv2.IMREAD_COLOR)
            img_HD = self.cvtcolor_and_resize(img_HD)

            if img_HD_prev is None:
                img_HD_prev = img_HD

            syneth, G_pred, skymask  = self.synthesize(img_HD, img_HD_prev)

            if self.save_jpgs:
                fpath = os.path.join(args.output_dir, img_names[idx])
                plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                plt.imsave(fpath[:-4] + 'coarse_skymask.jpg', G_pred)
                plt.imsave(fpath[:-4] + 'refined_skymask.jpg', skymask)
                plt.imsave(fpath[:-4] + 'syneth.jpg', syneth.clip(min=0, max=1))

            self.write_video(img_HD, syneth)
            print('processing: %d / %d ...' % (idx, len(img_names)))

            img_HD_prev = img_HD




    def run_video(self):

        print('running evaluation...')

        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_HD_prev = None

        idx = 0

        while (1):
            ret, frame = cap.read()
            if ret:
                img_HD = self.cvtcolor_and_resize(frame)

                if img_HD_prev is None:
                    img_HD_prev = img_HD

                syneth, G_pred, skymask = self.synthesize(img_HD, img_HD_prev)

                if self.save_jpgs:
                    fpath = os.path.join(args.output_dir, str(idx)+'.jpg')
                    plt.imsave(fpath[:-4] + '_input.jpg', img_HD)
                    plt.imsave(fpath[:-4] + '_coarse_skymask.jpg', G_pred)
                    plt.imsave(fpath[:-4] + '_refined_skymask.jpg', skymask)
                    plt.imsave(fpath[:-4] + '_syneth.jpg', syneth.clip(min=0, max=1))

                self.write_video(img_HD, syneth)
                print('processing: %d / %d ...' % (idx, m_frames))

                img_HD_prev = img_HD
                idx += 1

            else:  # if reach the last frame
                break


    def run(self):
        if self.input_mode == 'seq':
            print('sequence mode')
            self.run_imgseq()
        # elif self.input_mode == 'video':
        #     print('video mode')
        #     self.run_video()
        elif self.input_mode == 'video':
            if hasattr(self, 'run_video_multithreaded'):
                self.run_video_multithreaded()
            else:
                self.run_video()
        elif self.input_mode == 'webcam':
            self.run_webcam()
        else:
            print('wrong input_mode, select one in [seq, video')
            exit()

class SkyFilterMultithreaded(SkyFilter):
    def __init__(self, args):
        super().__init__(args)
        # self.net_G = torch.compile(self.net_G)  # 加速模型
        self.num_processor_threads = 4  # 可调节并发线程数量
        self.batch_size = 4  # 每次推理处理帧数
        self.lock = threading.Lock()

    def run_video_multithreaded(self):
        print('running multithreaded video evaluation...')

        cap = cv2.VideoCapture(self.datadir)
        self.m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.read_queue = queue.Queue(maxsize=30)
        self.write_queue = queue.Queue(maxsize=30)

        self.active_processors = self.num_processor_threads
        self.stop_signal = False

        # 启动线程
        threading.Thread(target=self.read_frames, args=(cap,), daemon=True).start()
        for _ in range(self.num_processor_threads):
            threading.Thread(target=self.process_frames, daemon=True).start()
        threading.Thread(target=self.write_frames, daemon=True).start()

        # 等待直到处理完成
        while not self.stop_signal:
            if self.read_queue.empty() and self.write_queue.empty() and self.active_processors == 0:
                break

        cap.release()
        self.video_writer.release()
        self.video_writer_cat.release()
        cv2.destroyAllWindows()

    def read_frames(self, cap):
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                for _ in range(self.num_processor_threads):
                    self.read_queue.put(None)
                break
            self.read_queue.put((idx, frame))
            idx += 1

    def process_frames(self):
        while True:
            batch = []
            idx_batch = []
            img_HD_batch = []
            img_HD_prev_batch = []

            for _ in range(self.batch_size):
                item = self.read_queue.get()
                if item is None:
                    break
                idx, frame = item
                img_HD = self.cvtcolor_and_resize(frame)
                img_HD_batch.append(img_HD)
                idx_batch.append(idx)

            if not img_HD_batch:
                with self.lock:
                    self.active_processors -= 1
                    if self.active_processors == 0:
                        self.write_queue.put(None)
                break

            for i in range(len(img_HD_batch)):
                img_HD_prev_batch.append(img_HD_batch[i - 1] if i > 0 else img_HD_batch[0])

            results = self.synthesize_batch(img_HD_batch, img_HD_prev_batch)

            for i in range(len(results)):
                syneth, G_pred, skymask = results[i]
                self.write_queue.put((idx_batch[i], img_HD_batch[i], syneth))

    def write_frames(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.stop_signal = True
                break

            idx, img_HD, syneth = item
            self.write_video(img_HD, syneth)
            print(f'Processed frame {idx} / {self.m_frames}')

    def write_video(self, img_HD, syneth):
        frame = np.array(255.0 * syneth[:, :, ::-1], dtype=np.uint8)
        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = np.array(255.0 * frame_cat[:, :, ::-1], dtype=np.uint8)

        with self.lock:
            self.video_writer.write(frame)
            self.video_writer_cat.write(frame_cat)
            # cv2.imshow('frame_cat', frame_cat)
            # cv2.waitKey(1)

    def synthesize_batch(self, imgs_HD, imgs_HD_prev):
        h, w, c = imgs_HD[0].shape

        imgs_tensor = []
        for img_HD in imgs_HD:
            img = cv2.resize(img_HD, (self.in_size_w, self.in_size_h))
            img = np.array(img, dtype=np.float32)
            img = torch.tensor(img).permute([2, 0, 1]).unsqueeze(0)
            imgs_tensor.append(img)

        imgs_tensor = torch.cat(imgs_tensor, dim=0).to(device)

        with torch.no_grad():
            G_preds = self.net_G(imgs_tensor)
            G_preds = torch.nn.functional.interpolate(G_preds, (h, w), mode='bicubic', align_corners=False)

        results = []
        for i in range(G_preds.shape[0]):
            G_pred = G_preds[i].permute(1, 2, 0)
            G_pred = torch.cat([G_pred] * 3, dim=-1)
            G_pred = np.clip(G_pred.cpu().numpy(), 0, 1)
            skymask = self.skyboxengine.skymask_refinement(G_pred, imgs_HD[i])
            syneth = self.skyboxengine.skyblend(imgs_HD[i], imgs_HD_prev[i], skymask)
            results.append((syneth, G_pred, skymask))

        return results


if __name__ == '__main__':

    config_path = parser.parse_args().path
    args = utils.parse_config(config_path)
    sf = SkyFilterMultithreaded(args)
    # sf.net_G = torch.compile(sf.net_G)
    # print('model compiled')
    # sf = SkyFilter(args)
    T1 = time.time()
    sf.run()
    T2 = time.time()
    print('Time: %f' % (T2 - T1))


