#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

# 运行./cmd.sh。目的是设置队列的类型。queue.pl/slurm.pl：执行程序需要排队；run.pl：无需排队（但可能会消耗大量内存）。
. ./cmd.sh
# 运行./path.sh。目的是导入kaldi库的根目录以及其他目录。
. ./path.sh
# Linux里面set-e命令作用是，如果一个命令返回一个非0退出状态值(失败)，就退出，不会继续执行。
set -e
# Mel频率倒谱系数(Mel Frequency Cepstrum Coefficient,MFCC)
# mfcc文件的存放路径
mfccdir=$(pwd)/mfcc
# VAD，语音活动检测(Voice Activity Detection,VAD)又称语音端点检测,语音边界检测。目的是从声音信号流里识别和消除长时间的静音期。
# vad文件的存放路径
vaddir=$(pwd)/mfcc

# 生成的数据对文件路径
voxceleb1_trials=data/voxceleb1_test/trials
# 数据所在文件目录
voxceleb1_root=data/
# 训练得到的模型存放路径
nnet_dir=exp/xvector_nnet_1a

# 阶段数
stage=0

if [ $stage -le 0 ]; then
  # 阶段0，根据数据集生成指定格式的测试数据
  local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test
fi

if [ $stage -le 1 ]; then
  # 阶段1，提出MFCC和VAD
  # Make MFCCs and compute the energy-based VAD for each dataset
  # 生成MFCC并计算每个数据集的基于能量的VAD
  # 梅尔倒谱系数（mfcc）
  # 提取过程：连续语音--预加重--加窗分帧--FFT--MEL滤波器组--对数运算--DCT
  for name in voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi

if [ $stage -le 9 ]; then
  # 阶段9，提取测试集的x-vectors
  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    $nnet_dir data/voxceleb1_test \
    $nnet_dir/xvectors_voxceleb1_test
fi

if [ $stage -le 11 ]; then
  # 阶段11，测试得到结果文件
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1
fi

if [ $stage -le 12 ]; then
  # 阶段12，根据结果文件计算EER
  eer=$(compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2>/dev/null)
  mindcf1=$(sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2>/dev/null)
  mindcf2=$(sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2>/dev/null)
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.224%
  # minDCF(p-target=0.01): 0.3492
  # minDCF(p-target=0.001): 0.5452
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.419%
  # minDCF(p-target=0.01): 0.4701
  # minDCF(p-target=0.001): 0.5981
fi
