# Amazon SageMaker Tensorflow ハンズオン

このハンズオンは [Amazon SageMaker](https://aws.amazon.com/jp/sagemaker/) 上で [Tensorflow](https://www.tensorflow.org/) を使った機械学習/深層学習を学ぶことを目的としています。

## 学べること
このコースを終えたあと、以下のような概念/方法を習得することができます。
1. Amazon SageMaker を使ってTensorFlow のモデルを構築・学習・デプロイする方法
1. Amazon SageMaker を使った分散学習 (複数CPU/GPU、あるいはマルチノード)
1. Amazon SageMaker Neo を使ったモデルコンパイル
1. Amazon SageMaker を使ったHPO最適化
1. Amazon SageMaker/Tensorflow を使ったTensorboardの使い方
1. Amazon SageMaker でKerasを使ったMNIST分類

なお、以下の知識を前提とします。
1. 機械学習/深層学習の概念と一般的な理解
1. Python/Tensorflow を用いたプログラミング
1. AWS の基礎的な知識と操作方法

## コンテンツ
これらのコンテンツを動かす SageMaker ノートブックインスタンスは `ml.c5.xlarge` を推奨します
1. TensorFlow のモデルを使った分散学習とNeoによるモデル最適化 [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_distributed_mnist_neo.ipynb)]
1. Tensorflow Containerを使ったHPO [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/hpo_tensorflow_mnist.ipynb)]
1. HPO ジョブの可視化 [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/HPO_Analyze_TuningJob_Results.ipynb)]
1. ResNet Cifer10によるTensorflowboardの利用[[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb)]
1. Keras (Tensorflow Backend) を利用してMNISTの分類を行う[[notebook](https://github.com/shokout/handson-201812/blob/master/keras_tensorflow/keras_tensorflow_mnist.ipynb)]


### 1. [分散学習](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_distributed_mnist_neo.ipynb "SAGEMAKER PYTHON SDK > tensorflow_distributed_mnist_neo.ipynb")

- 出力を見て複数ノードで学習が分散されていることを確認します。
- `hyperparameters={'epochs': 6}` でハイパーパラメータを渡すことができます。
- `SageMaker Neo`で推論モデルのランタイムをコンパイルし、推論時間がどのくらい短縮されるか試してみましょう。
- (optional) `metric_definitions` で CloudWatch メトリクスとして結果を出力することができます [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html)]。
    
    ```python
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\.]+)'},
    ]
    ```
- (optional) local mode: Local instance上での学習や推論が可能です。[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_distributed_mnist/tensorflow_local_mode_mnist.ipynb)

### 2. [ベイズ最適化 (HPO)](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/hpo_tensorflow_mnist.ipynb "HYPERPARAMETER TUNING > hpo_tensorflow_mnist.ipynb")
- SageMaker ではベイズ最適化を用いて、正規表現でパースされたメトリクスに対してハイパーパラメータの最適化 (HPO) を行うことができます。
-  HPO ジョブの結果を可視化しましょう [[notebook](https://github.com/shokout/handson-201812/blob/master/tensorflow_mnist_hpo/HPO_Analyze_TuningJob_Results.ipynb "HYPERPARAMETER TUNING > HPO_Analyze_TuningJob_Result.ipynb")]。
- (optional) 新たなパラメータを最適化対象として追加してみましょう。
    - `'batch_size': CategoricalParameter([50, 100, 200])` など。
- (optional) Warm Start を使って最適化ジョブを継続するよう書き換えてみましょう [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html), [参考ブログ](https://aws.amazon.com/jp/blogs/news/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/)]。
    - ```python
      from sagemaker.tuner import WarmStartConfig, WarmStartTypes
      hpo_warm_start_config = WarmStartConfig(WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
                                    parents={'<parent_tuning_job_name>','<parent_tuning_job_name_2>'})
      ```
    
    を実行し、 `warm_start_config=hpo_warm_start_config` を `HyperparameterTuner()` 作成時に追加。

### 3. [TensorBoardの利用方法](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/tensorflow_resnet_cifar10_with_tensorboard.ipynb "tensorflow_resnet_cifar10_with_tensorboard > tensorflow_resnet_cifar10_with_tensorboard.ipynb")
- SageMakerからTensorBoardを利用できます。
- Cifar10 を使った学習スクリプト [`source/cifar10.py`](https://github.com/shokout/handson-201812/blob/master/tensorflow_resnet_cifar10_with_tensorboard/source_dir/resnet_cifar_10.py) が用意されているので、これをエントリーポイントとした SageMaker の学習を行います。

### 4. [Keras on Tensorflow](https://github.com/shokout/handson-201812/blob/master/keras_tensorflow/keras_tensorflow_mnist.ipynb)
- Amazon SageMaker上でKerasを実行し、MNISTの手書き数字の分類を学習します。
- バックエンドはTensorflow。

