{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SageMaker PyTorch ハンズオン\n",
    "\n",
    "このハンズオンは [Amazon SageMaker](https://aws.amazon.com/jp/sagemaker/) 上で [PyTorch](https://pytorch.org/) を使った機械学習/深層学習を学ぶことを目的としています。\n",
    "\n",
    "## 学べること\n",
    "このコースを終えたあと、以下のような概念/方法を習得することができます。\n",
    "1. Amazon SageMaker を使ってTensorFlow のモデルを構築・学習・デプロイする方法\n",
    "1. Amazon SageMaker を使った分散学習 (複数CPU/GPU、あるいはマルチノード)\n",
    "1. Amazon SageMaker Neo を使ったモデルコンパイル\n",
    "1. Amazon SageMaker を使ったHPO最適化\n",
    "1. Amazon SageMaker/Tensorflow を使ったTensorboardの使い方\n",
    "\n",
    "なお、以下の知識を前提とします。\n",
    "1. 機械学習/深層学習の概念と一般的な理解\n",
    "1. Python/Tensorflow を用いたプログラミング\n",
    "1. AWS の基礎的な知識と操作方法\n",
    "\n",
    "## コンテンツ\n",
    "これらのコンテンツを動かす SageMaker ノートブックインスタンスは `ml.c5.xlarge` を推奨します\n",
    "1. TensorFlow のモデルを使った分散学習とNeoによるモデル最適化 [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_distributed_mnist_neo.ipynb)]\n",
    "1. Tensorflow Containerを使ったHPO [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/hpo_tensorflow_mnist.ipynb)]\n",
    "1. HPO ジョブの可視化 [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/HPO_Analyze_TuningJob_Results.ipynb)]\n",
    "1. ResNet Cifer10によるTensorflowboardの利用[[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb)]\n",
    "\n",
    "\n",
    "### 1. [分散学習](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_distributed_mnist_neo.ipynb \"SAGEMAKER PYTHON SDK > tensorflow_distributed_mnist_neo.ipynb\")\n",
    "\n",
    "- 出力を見て複数ノードで学習が分散されていることを確認します。\n",
    "- `hyperparameters={'epochs': 6}` でハイパーパラメータを渡すことができます。\n",
    "- `SageMaker Neo`で推論モデルのランタイムをコンパイルし、推論時間がどのくらい短縮されるか試してみましょう。\n",
    "- (optional) `metric_definitions` で CloudWatch メトリクスとして結果を出力することができます [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html)]。\n",
    "    \n",
    "    ```python\n",
    "    metric_definitions=[\n",
    "        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\\.]+)'},\n",
    "    ]\n",
    "    ```\n",
    "- (optional) local mode: Local instance上での学習や推論が可能です。[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb)\n",
    "\n",
    "### 2. [ベイズ最適化 (HPO)](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/hpo_tensorflow_mnist.ipynb \"HYPERPARAMETER TUNING > hpo_tensorflow_mnist.ipynb\")\n",
    "- SageMaker ではベイズ最適化を用いて、正規表現でパースされたメトリクスに対してハイパーパラメータの最適化 (HPO) を行うことができます。\n",
    "-  HPO ジョブの結果を可視化しましょう [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/HPO_Analyze_TuningJob_Results.ipynb \"HYPERPARAMETER TUNING > HPO_Analyze_TuningJob_Result.ipynb\")]。\n",
    "- (optional) 新たなパラメータを最適化対象として追加してみましょう。\n",
    "    - `'batch_size': CategoricalParameter([50, 100, 200])` など。\n",
    "- (optional) Warm Start を使って最適化ジョブを継続するよう書き換えてみましょう [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html), [参考ブログ](https://aws.amazon.com/jp/blogs/news/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/)]。\n",
    "    - ```python\n",
    "      from sagemaker.tuner import WarmStartConfig, WarmStartTypes\n",
    "      hpo_warm_start_config = WarmStartConfig(WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,\n",
    "                                    parents={'<parent_tuning_job_name>','<parent_tuning_job_name_2>'})\n",
    "      ```\n",
    "    \n",
    "    を実行し、 `warm_start_config=hpo_warm_start_config` を `HyperparameterTuner()` 作成時に追加。\n",
    "\n",
    "### 3. [Tensorflow TensorBoard](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb \"tensorflow_resnet_cifar10_with_tensorboard > tensorflow_resnet_cifar10_with_tensorboard.ipynb\")\n",
    "- SageMakerからTensorBoardを利用できます。\n",
    "- Cifar10 を使った学習スクリプト [`source/cifar10.py`](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/source_dir/resnet_cifar_10.py) が用意されているので、これをエントリーポイントとした SageMaker の学習を行います。\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "conda_pytorch_p36",
   "language": "python",
   "name": "conda_pytorch_p36"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
