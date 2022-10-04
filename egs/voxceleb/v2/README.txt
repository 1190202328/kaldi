
 UPLOADER        David Snyder 
 DATE            2018-05-30
 KALDI VERSION   b1be44e

 This recipe replaces ivectors used in the v1 recipe with embeddings extracted
 from a deep neural network.  In the scripts, we refer to these embeddings as
 "xvectors."  The recipe is closely based on the following paper:
 http://www.danielpovey.com/files/2018_icassp_xvectors.pdf but is trained
 exclusively on wideband data (augmented VoxCeleb 1 and 2).

# 该配方用从深层神经网络提取的embedding替换了v1配方中使用的ivectors。在脚本中，我们将这些嵌入称为“xvectors”。该配方密切基于以下论文：http://www.danielpovey.com/files/2018_icassp_xvectors.pdf，但专门针对wideband data（增强的VoxCeleb 1和2）进行训练。

 This directory contains files generated from the recipe in egs/voxceleb/v2/.
 It's contents should be placed in a similar directory, with symbolic links to
 utils/, sid/, steps/, etc.  This was created when Kaldi's master branch was
 at git log b1be44eb2ac86a8cf0346e1da86abbcfd7d5251e.
# 此目录包含从egs/voxceleb/v2/中的配方生成的文件。它的内容应该放在类似的目录中，带有指向utils/、sid/、steps/等的符号链接。这是在Kaldi的主分支位于git log b1be44eb2ac86a8cf0346e1da86abbcfd7d5251e时创建的。


 I. Files list
 ------------------------------------------------------------------------------
 
 ./
     README.txt               This file
     run.sh                   A copy of the egs/voxceleb/v2/run.sh
                              at the time of uploading this file.  Look at this
                              to see examples of computing features and 
                              extracting x-vectors.
				# egs/voxceleb/v2/run的副本。在上载此文件时显示。看看这里，可以看到计算特征和提取x矢量的示例。

 local/nnet3/xvector/tuning/
     run_xvector_1a.sh        This is the default recipe, at the time of
                              uploading this resource.  The script generates
                              the configs, egs, and trains the model.
				# 这是上载此资源时的默认配方。该脚本生成配置、egs并训练模型。

 conf/
     vad.conf                 The energy-based VAD configuration # 基于能量的VAD配置
     mfcc.conf                A wideband MFCC configuration # wideband MFCC配置

 exp/xvector_nnet_1a/ 
     final.raw                The pretrained DNN model # 预训练的DNN模型
     nnet.config              The nnet3 config file that was used when the
                              DNN model was first instantiated.
				# 首次实例化DNN模型时使用的nnet3配置文件。
     extract.config           Another nnet3 config file that modifies the DNN
                              final.raw to extract x-vectors.  It should be
                              automatically handled by the script 
                              extract_xvectors.sh.
				# 另一个修改DNN final.raw的nnet3配置文件以提取x-vector。它应该由脚本extract_xvectors.sh自动处理。
     min_chunk_size           Min chunk size used (see extract_xvectors.sh) # 使用的最小块大小（请参阅extract_xvectors.sh）
     max_chunk_size           Max chunk size used (see extract_xvectors.sh) # 使用的最大块大小（请参阅extract_xvectors.sh）
     srand                    The RNG seed used when creating the DNN # 随机数种子

 exp/xvector_nnet_1a/xvectors_train/
     mean.vec                 Vector for centering # 中心化向量
     transform.mat            Whitening matrix # 增白矩阵
     plda                     PLDA model # PLDA模型

 II. Citation
 ------------------------------------------------------------------------------

 If you wish to use this system in a publication, please cite
 "X-vectors: Robust DNN Embeddings for Speaker Recognition."  The
 recipe is closely based on that paper.  The main difference is that
 here we use exclusively wideband training data.  The bibtex is as follows:

 @inproceedings{snyder2018xvector,
 title={X-vectors: Robust DNN Embeddings for Speaker Recognition},
 author={Snyder, D. and Garcia-Romero, D. and Sell, G. and Povey, D. and Khudanpur, S.},
 booktitle={2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 year={2018},
 organization={IEEE},
 url={http://www.danielpovey.com/files/2018_icassp_xvectors.pdf}
 }

 To refer to the VoxCeleb 1 and 2 datasets in a publication,
 navigate to the following URLs:
     http://www.robots.ox.ac.uk/~vgg/data/voxceleb
     http://www.robots.ox.ac.uk/~vgg/data/voxceleb2

 III. Corpora
 ------------------------------------------------------------------------------

 The pretrained model used the following datasets for training.
     
     VoxCeleb 1 Dev          http://www.robots.ox.ac.uk/~vgg/data/voxceleb
     VoxCeleb 2              http://www.robots.ox.ac.uk/~vgg/data/voxceleb2
     MUSAN                   http://www.openslr.org/17
     RIR_NOISES              http://www.openslr.org/28

 The following dataset was used for evaluation.

     VoxCeleb 1 Test         http://www.robots.ox.ac.uk/~vgg/data/sitw
