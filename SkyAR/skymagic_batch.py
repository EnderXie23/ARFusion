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
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from queue import Queue

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
        self.video_writer = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (args.out_size_w, args.out_size_h))
        self.video_writer_cat = cv2.VideoWriter('demo-cat.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                            20.0, (2*args.out_size_w, args.out_size_h))
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
            self.run_video_batch()
            # self.run_video()
        elif self.input_mode == 'webcam':
            self.run_webcam()
        else:
            print('wrong input_mode, select one in [seq, video]')
            exit()

class DualVideoWriterThreaded:
    def __init__(self, path1, path2, size1, size2, fps=20.0, codec='MP4V'):
        self.queue = Queue()
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer1 = cv2.VideoWriter(path1, fourcc, fps, size1)
        self.writer2 = cv2.VideoWriter(path2, fourcc, fps, size2)
        self.size1 = size1
        self.size2 = size2
        self.thread = threading.Thread(target=self._worker)
        self.thread.start()
    
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            frame1, frame2 = item
            self.writer1.write(frame1)
            self.writer2.write(frame2)
            self.queue.task_done()

    def write(self, frame, frame_cat):
        self.queue.put((frame, frame_cat))

    def close(self):
        self.queue.put(None)
        self.thread.join()
        self.writer1.release()
        self.writer2.release()

class SkyFilterBatched(SkyFilter):
    def __init__(self, args):
        super().__init__(args)
        self.dual_video_writer = DualVideoWriterThreaded(
            path1='demo.mp4',
            path2='demo-cat.mp4',
            size1=(self.out_size_w, self.out_size_h),  # 注意这里是 (width, height)
            size2=(2 * self.out_size_w, self.out_size_h),  # 注意这里是 (width, height)
        )

    def run_video_batch(self):
        print('running batched video processing...')

        cap = cv2.VideoCapture(self.datadir)
        m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        BATCH_SIZE = 16

        while True:
            t1 = time.time()
            img_HD_batch = []
            idx_batch = []

            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break

                img_HD = self.cvtcolor_and_resize(frame)
                img_HD_batch.append(img_HD)
                idx_batch.append(idx)
                idx += 1

            if not img_HD_batch:
                break

            # 使用上一帧，或用自己代替
            img_HD_prev_batch = [img_HD_batch[i - 1] if i > 0 else img_HD_batch[0] for i in range(len(img_HD_batch))]
            t2 = time.time()
            print(f'Preprocessing time: {t2 - t1:.4f} seconds')

            results = self.synthesize_batch(img_HD_batch, img_HD_prev_batch)
            t3 = time.time()
            print(f'Processing time for batch: {t3 - t2:.4f} seconds')

            for i, (syneth, G_pred, skymask) in enumerate(results):
                self.write_video(img_HD_batch[i], syneth)
                print(f'Processed frame {idx_batch[i]} / {m_frames}')
            t4 = time.time()
            print(f'Writing time: {t4 - t3:.4f} seconds')

        cap.release()
        self.dual_video_writer.close()
        # cv2.destroyAllWindows()

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

    def write_video(self, img_HD, syneth):
        frame = (syneth[:, :, ::-1] * 255).astype(np.uint8)
        frame_cat = np.concatenate([img_HD, syneth], axis=1)
        frame_cat = (frame_cat[:, :, ::-1] * 255).astype(np.uint8)

        self.dual_video_writer.write(frame, frame_cat)


if __name__ == '__main__':
    config_path = parser.parse_args().path
    args = utils.parse_config(config_path)
    sf = SkyFilterBatched(args)
    # sf = SkyFilter(args)
    T1 = time.time()
    sf.run()
    T2 = time.time()
    print('Time: %f' % (T2 - T1))
