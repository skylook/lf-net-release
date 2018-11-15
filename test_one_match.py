from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
# from tqdm import tqdm
import pickle
import time

from datasets import *

from det_tools import *
from eval_tools import compute_sift, compute_sift_multi_scale, draw_match, draw_keypoints, draw_match2
from common.tf_train_utils import get_optimizer
# from imageio import imread, imsave
from inference import *


MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def build_networks(config, photo, is_training):

    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)

    # extract patches
    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    # Descriptor
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False) # [B*K,D]

    # scale and orientation (extra)
    scale_maps = det_endpoints['scale_maps']
    ori_maps = det_endpoints['ori_maps'] # cos/sin
    degree_maps, _ = get_degree_maps(ori_maps) # degree (rgb psuedo color code)
    kpts_scale = det_endpoints['kpts_scale']
    kpts_ori = det_endpoints['kpts_ori']
    kpts_ori = tf.atan2(kpts_ori[:,1], kpts_ori[:,0]) # radian

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
        # EXTRA
        'scale_maps': scale_maps,
        'kpts_scale': kpts_scale,
        'degree_maps': degree_maps,
        'kpts_ori': kpts_ori,
    }

    return ops

def detect_compute(photo, sess, ops, photo_ph, config):
    height, width = photo.shape[:2]

    print('photo.shape = {}'.format(photo.shape))

    longer_edge = max(height, width)
    if config.max_longer_edge > 0 and longer_edge > config.max_longer_edge:
        if height > width:
            new_height = config.max_longer_edge
            new_width = int(width * config.max_longer_edge / height)
        else:
            new_height = int(height * config.max_longer_edge / width)
            new_width = config.max_longer_edge
        photo = cv2.resize(photo, (new_width, new_height))
        height, width = photo.shape[:2]
    # rgb = photo.copy()
    if photo.ndim == 3 and photo.shape[-1] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    photo = photo[None,...,None].astype(np.float32) / 255.0 # normalize 0-1
    assert photo.ndim == 4 # [1,H,W,1]

    feed_dict = {
        photo_ph: photo,
    }
    if config.full_output:
        fetch_dict = {
            'kpts': ops['kpts'],
            'feats': ops['feats'],
            'kpts_scale': ops['kpts_scale'],
            'kpts_ori': ops['kpts_ori'],
            'scale_maps': ops['scale_maps'],
            'degree_maps': ops['degree_maps'],
        }
        # outs = sess.run(fetch_dict, feed_dict=feed_dict)
        # draw key
        # kp_img = draw_keypoints(rgb, outs['kpts'])
        # scale_range = config.net_max_scale-config.net_min_scale
        # if scale_range == 0:
        #     scale_range = 1.0
        # scale_img = (outs['scale_maps'][0]*255/scale_range).astype(np.uint8)
        # ori_img = (outs['degree_maps'][0]*255).astype(np.uint8)

        # out_img_path = os.path.join(config.out_dir, os.path.basename(img_path))
        # imsave(out_img_path, kp_img)
        # imsave(out_img_path+'-scl.jpg', scale_img)
        # imsave(out_img_path+'-ori.jpg', ori_img)
        # cv2.imwrite(out_img_path, kp_img)
        # cv2.imwrite(out_img_path+'-scl.jpg', scale_img)
        # cv2.imwrite(out_img_path+'-ori.jpg', ori_img)
        # np.savez(out_img_path+'.npz', kpts=outs['kpts'], descs=outs['feats'], size=np.array([height, width]),
        #          scales=outs['kpts_scale'], oris=outs['kpts_ori'])
    else:
        # Dump keypoint locations and their features
        fetch_dict = {
            'kpts': ops['kpts'],
            'feats': ops['feats'],
        }
        # out_path = os.path.join(config.out_dir, os.path.basename(img_path)+'.npz')
        # np.savez(out_path, kpts=outs['kpts'], feats=outs['feats'], size=np.array([height, width]))

    outs = sess.run(fetch_dict, feed_dict=feed_dict)

    return outs

def cross_check(matches12, matches21):
    d21 = {}
    for m21 in matches21:
        d21[m21.queryIdx] = m21.trainIdx

    # print('d21 = {}'.format(d21))

    matches = []
    for m12 in matches12:
        if m12.queryIdx == d21[m12.trainIdx]:
            matches.append(m12)

    return matches

def match(query_feat, train_feat, threshold):
    # print('kpts1 = {}'.format(kpts1[0]))
    # print('desc_feats1.shape = {}'.format(desc_feats1[0].shape))
    # print('desc_feats1 = {}'.format(desc_feats1[0]))

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors.
    # matches = bf.match(query_feat, train_feat)

    matches = bf.knnMatch(query_feat, train_feat, k=2)

    # print('matches[0] = {}'.format(matches[0]))

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance <= threshold and m.distance < 0.8 * n.distance:
            good_matches.append(m)
    # for m, n in matches:
    #     if m.distance <= threshold:
    #         good_matches.append(m)
    # print('good_matches[0] = {}'.format(good_matches[0]))
    # Sort them in the order of their distance.
    good_matches12 = sorted(good_matches, key=lambda x: x.distance)

    # cross check
    matches21 = bf.match(train_feat, query_feat)

    final_matches = cross_check(good_matches12, matches21)

    print('final matches = {}'.format(final_matches))

    # return nn_dist, nn_inds2
    return final_matches

def draw_matches(query_image, query_pts, train_image, train_pts, matches):
    height, width = query_image.shape[:2]
    canvas = np.concatenate([train_image, query_image], axis=1)

    print('canvas.shape[-1] = {}'.format(canvas.shape[-1]))

    if canvas.shape[-1] == 1:
        canvas = np.repeat(canvas, 3, -1) # gray to rgb

    for m in matches:
        # print('m.queryIdx = {}, m.trainIdx = {}'.format(m.queryIdx, m.trainIdx))
        queryIdx = m.queryIdx
        trainIdx = m.trainIdx
        # queryIdx = m.trainIdx
        # trainIdx = m.queryIdx
        x1,y1 = train_pts[trainIdx]
        x2,y2 = query_pts[queryIdx]
        x1 = int(x1+0.5)
        y1 = int(y1+0.5)
        x2 = int(x2+0.5)
        y2 = int(y2+0.5)

        # color = tuple(np.round(np.random.rand(3) * 255).astype(np.uint8))
        # rnd = np.round(np.random.rand(3) * 255).astype(np.int32)
        # color = (rnd[0], rnd[1], rnd[2])
        # color = (251, 62, 63)
        # color = (0,0xFF,0)
        # color = list(np.random.choice(range(256), size=3))
        # color = np.round(np.random.rand(3) * 255).astype(np.int32)
        color1 = np.uint8(np.random.uniform(0, 255, 3))

        color = tuple(map(int, color1))

        # print('color = {}'.format(color))
        cv2.line(canvas, (x1, y1), (x2 + width, y2), color, 1)
        cv2.circle(canvas, (x1, y1), 2, color)
        cv2.circle(canvas, (x2 + width, y2), 2, color)

    return canvas

def dump_debug(name, photo, outs, config):
    height, width = photo.shape[:2]

    # draw key
    kp_img = draw_keypoints(photo, outs['kpts'])
    scale_range = config.net_max_scale-config.net_min_scale
    if scale_range == 0:
        scale_range = 1.0

    base_name = os.path.basename(name)

    out_img_path = os.path.join(config.out_dir, name)
    out_npz_path = os.path.join(config.out_dir, base_name +'.npz')

    if config.full_output:
        scale_img = (outs['scale_maps'][0]*255/scale_range).astype(np.uint8)
        ori_img = (outs['degree_maps'][0]*255).astype(np.uint8)


        # imsave(out_img_path, kp_img)
        # imsave(out_img_path+'-scl.jpg', scale_img)
        # imsave(out_img_path+'-ori.jpg', ori_img)
        cv2.imwrite(out_img_path, kp_img)
        cv2.imwrite(out_img_path+'-scl.jpg', scale_img)
        cv2.imwrite(out_img_path+'-ori.jpg', ori_img)

        np.savez(out_npz_path, kpts=outs['kpts'], descs=outs['feats'], size=np.array([height, width]),
                 scales=outs['kpts_scale'], oris=outs['kpts_ori'])
    else:
        np.savez(out_npz_path, kpts=outs['kpts'], feats=outs['feats'], size=np.array([height, width]))

def print_matches(matches):
    idx = 0
    for m in matches:
        print('matches[{}] = {}, {}, {}'.format(idx, m.queryIdx, m.trainIdx, m.distance))
        idx+=1

    return

def main(config):

    # Build Networks
    tf.reset_default_graph()

    is_training = tf.constant(False) # Always False in testing

    photo_ph = tf.placeholder(tf.float32, [1, None, None, 1]) # input grayscale image, normalized by 0~1
    ops = build_networks(config, photo_ph, is_training)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True 
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())

    # 加载参数
    saver = tf.train.Saver()
    print('Load trained models...')

    if os.path.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
        model_dir = config.model
    else:
        checkpoint = config.model
        model_dir = os.path.dirname(config.model)


    if checkpoint is not None:
        print('Checkpoint', os.path.basename(checkpoint))
        print("[{}] Resuming...".format(time.asctime()))
        saver.restore(sess, checkpoint)
    else:
        raise ValueError('Cannot load model from {}'.format(model_dir))    
    print('Done.')

    # Ready to feed input images
    # img_paths = [x.path for x in os.scandir(config.in_dir) if x.name.endswith('.jpg') or x.name.endswith('.png')]

    train_path = config.train
    query_path = config.query
    print('train_path = {}'.format(train_path))
    print('query_path = {}'.format(query_path))

    # if not os.path.exists(config.out_dir):
    #     os.makedirs(config.out_dir)

    avg_elapsed_time = 0

    # 读取和处理图片
    train_img = cv2.imread(train_path, cv2.IMREAD_COLOR)
    query_img = cv2.imread(query_path, cv2.IMREAD_COLOR)

    # for img_path in tqdm(img_paths):
    # photo = imread(img_path)
    # photo = cv2.imread(img_path, cv2.IMREAD_COLOR)

    start_time = time.time()  # 记录代码开始时间

    query_out = detect_compute(photo=query_img, sess=sess, ops=ops, photo_ph=photo_ph, config=config)
    train_out = detect_compute(photo=train_img, sess=sess, ops=ops, photo_ph=photo_ph, config=config)

    matches = match(query_feat=query_out['feats'], train_feat=train_out['feats'], threshold=1.0)

    end_time = time.time()  # 记录代码结束时间
    run_time = end_time - start_time  # 计算运行时间
    print('run_time: ', run_time)

    print_matches(matches[:10])

    # def draw_matches(img1, img2, kpts1, kpts2, matches)
    outImg = draw_matches(query_image=query_img, query_pts=query_out['kpts'],
                          train_image=train_img, train_pts=train_out['kpts'],
                          matches=matches[:100])

    cv2.imwrite('matches.png', outImg)

    # img3 = cv2.drawMatches(query_img, outs1['kpts'], train_img, outs2['kpts'], matches[:10], None, flags=2)
    # cv2.drawMatches(query_img, outs1['kpts'], train_img, outs2['kpts'], matches[:10], outImg)
    #
    # cv2.imshow('matches', img3)
    # cv2.waitKey(0)

    # plt.imshow(img3), plt.show()
    # 绘制和输出
    # dump_debug(name='debug.png', photo=query_img, outs=outs, config=config)

    # 完成

    print('Done...')

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    io_arg = add_argument_group('In/Out', parser)
    io_arg.add_argument('--train', type=str, default='./test_1/daytime2.jpg',
                            help='original image')
    io_arg.add_argument('--query', type=str, default='./test_1/nights2.jpg',
                            help='query image')
    io_arg.add_argument('--out_dir', type=str, default='./dump_feats',
                            help='where to save keypoints')
    io_arg.add_argument('--full_output', type=str2bool, default=False,
                            help='dump keypoint image')

    model_arg = add_argument_group('Model', parser)
    model_arg.add_argument('--model', type=str, default='./release/models/outdoor/',
                            help='model file or directory')
    model_arg.add_argument('--top_k', type=int, default=500,
                            help='number of keypoints')
    model_arg.add_argument('--max_longer_edge', type=int, default=640,
                            help='resize image (do nothing if max_longer_edge <= 0)')

    tmp_config, unparsed = get_config(parser)

    # print('tmp_config = {}'.format(tmp_config))

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    # restore other hyperparams to build model
    if os.path.isdir(tmp_config.model):
        config_path = os.path.join(tmp_config.model, 'config.pkl')
    else:
        config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
    try:
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
    except:
        raise ValueError('Fail to open {}'.format(config_path))

    for attr, dst_val in sorted(vars(tmp_config).items()):
        if hasattr(config, attr):
            src_val = getattr(config, attr)
            if src_val != dst_val:
                setattr(config, attr, dst_val)
        else:
            # print('attr = {} dst_val = {}'.format(attr, dst_val))

            setattr(config, attr, dst_val)

    # print('config = {}'.format(config))

    main(config)