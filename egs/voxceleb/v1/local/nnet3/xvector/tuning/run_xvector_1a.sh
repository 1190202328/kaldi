#!/usr/bin/env bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#
# Copied from egs/sre16/v1/local/nnet3/xvector/tuning/run_xvector_1a.sh (commit e082c17d4a8f8a791428ae4d9f7ceb776aef3f0b).
#
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2018_icassp_xvectors.pdf

# 运行./cmd.sh。目的是设置队列的类型。queue.pl/slurm.pl：执行程序需要排队；run.pl：无需排队（但可能会消耗大量内存）。
. ./cmd.sh
# Linux里面set-e命令作用是，如果一个命令返回一个非0退出状态值(失败)，就退出，不会继续执行。
set -e

# 阶段数
stage=1
# 训练的阶段
train_stage=0
# 是否使用gpu
use_gpu=true
remove_egs=false

# 数据所在文件夹
data=data/train
#
nnet_dir=exp/xvector_nnet_1a/
egs_dir=exp/xvector_nnet_1a/egs

# 运行./path.sh。目的是导入kaldi库的根目录以及其他目录。
. ./path.sh
# 运行./cmd.sh。目的是设置队列的类型。queue.pl/slurm.pl：执行程序需要排队；run.pl：无需排队（但可能会消耗大量内存）。
. ./cmd.sh
#
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
# 现在，我们使用sid/net3/xvector/get_egs.sh创建nnet示例。
# 参数--num repeats与说话人在每个存档中重复的次数有关。
# 如果您似乎获得了太多的归档文件（例如，超过200个），请尝试增加--framesperiter选项。
# 参数--min frames per chunk和--max frames perchunk指定示例中特征的最小和最大长度（根据帧数）。
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23
# 为了理解egs脚本，可能需要在第3阶段之后立即放置一个“exit 1”命令。
# 然后，检查exp/<your dir>/egs/temp/ranges.*。
# 范围中的文件指定将创建的示例，以及这些示例将存储在哪些存档中。每行.*具有以下形式：

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
# 如果您对存档的数量（例如，50-150个存档是合理的）和每个演讲者的示例数量（例如1000-5000个是合理的”）感到满意，那么您可以让脚本继续到后面的阶段。
# 否则，请尝试增加或减少--num repeats选项。
# 您可能需要处理--framesperiter。增加此值会减少存档的数量，并增加每个存档的示例数。
# 减小此值会增加存档的数量，同时减少每个存档的示例数。
if [ $stage -le 6 ]; then
  # 阶段6，
  echo "$0: Getting neural network training egs"
  # dump egs.
  # 存储 egs
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{03,04,05,06}/$USER/kaldi-data/egs/voxceleb2/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --num-repeats 50 \
    "$data" $egs_dir
fi

if [ $stage -le 7 ]; then
  # 阶段7，
  echo "$0: creating neural net configs using the xconfig parser"
  num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
  feat_dim=$(cat $egs_dir/info/feat_dim)

  # This chunk-size corresponds to the maximum number of frames the
  # stats layer is able to pool over.  In this script, it corresponds
  # to 100 seconds.  If the input recording is greater than 100 seconds,
  # we will compute multiple xvectors from the same recording and average
  # to produce the final xvector.
  # 该chunk-size对应于stats层能够池化的最大帧数。在这个脚本中，它对应于100秒。
  # 如果输入记录大于100秒，我们将从同一记录中计算多个xvectors并求平均值，以生成最终的xvector。
  max_chunk_size=10000

  # The smallest number of frames we're comfortable computing an xvector from.
  # Note that the hard minimum is given by the left and right context of the
  # frame-level layers.
  # 计算xvector所需的最小帧数。请注意，硬最小值由frame-level的左右上下文给出。
  min_chunk_size=25
  mkdir -p $nnet_dir/configs
  cat <<EOF >$nnet_dir/configs/network.xconfig
  # please note that it is important to have input layer with the name=input

  # The frame-level layers
  input dim=${feat_dim} name=input
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
  relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn4 dim=512
  relu-batchnorm-layer name=tdnn5 dim=1500

  # The stats pooling layer. Layers after this are segment-level.
  # In the config below, the first and last argument (0, and ${max_chunk_size})
  # means that we pool over an input segment starting at frame 0
  # and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
  # mean that no subsampling is performed.
  stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

  # This is where we usually extract the embedding (aka xvector) from.
  relu-batchnorm-layer name=tdnn6 dim=512 input=stats

  # This is where another layer the embedding could be extracted
  # from, but usually the previous one works better.
  relu-batchnorm-layer name=tdnn7 dim=512
  output-layer name=output include-log-softmax=true dim=${num_targets}
EOF

  steps/nnet3/xconfig_to_configs.py \
    --xconfig-file $nnet_dir/configs/network.xconfig \
    --config-dir $nnet_dir/configs/
  cp $nnet_dir/configs/final.config $nnet_dir/nnet.config

  # These three files will be used by sid/nnet3/xvector/extract_xvectors.sh
  # 这三个文件将由sid/nnet3/xvector/extract_xvectors.sh使用
  echo "output-node name=output input=tdnn6.affine" >$nnet_dir/extract.config
  echo "$max_chunk_size" >$nnet_dir/max_chunk_size
  echo "$min_chunk_size" >$nnet_dir/min_chunk_size
fi

dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -le 8 ]; then
  # 阶段8，
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=3 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=true \
    --dir=$nnet_dir || exit 1
fi

exit 0
