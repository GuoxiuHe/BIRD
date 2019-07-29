# BIRD
SIGIR 2019: Finding Camouflaged Needle in a Haystack? Pornographic Products Detection via Berrypicking Tree Model

## Introduction

This is an implementation of Berry PIcking TRee MoDel (BIRD) based on Tensorflow.

## Experimental Environments

* Linux: RHEL 7
* Python: 3.6.6
* Tensorflow: 1.12.0
* GPU: Tesla P100-PCIE-16GB
* NVIDIA Driver Version: 396.44
* CUDA: 9.0
* CUDNN: 7.4.1

## The Structure of the Folder

```
BIRD
├── Dataset
│   ├── dictionary
│   │   └── PPDD
│   │       └── word_index.json
│   ├── PPDD
│   │   ├── data_list.txt
│   │   ├── online_test_1.txt
│   │   ├── online_test_2.txt
│   │   ├── README
│   │   ├── test_list.txt
│   │   ├── train_list.txt
│   │   └── val_list.txt
│   └── sample
│       └── sample.txt
├── README.md
├── Scripts
│   ├── config
│   │   ├── data
│   │   │   └── config.ppdd.json
│   │   └── model
│   │       └── config.bird.json
│   ├── logs
│   │   ├── bird.ppdd.evaluate_test.Sun_May_.log
│   │   ├── bird.ppdd.evaluate_val.Sun_May_.log
│   │   ├── bird.ppdd.predict_items2.Sun_May_.log
│   │   └── bird.ppdd.predict_items.Sun_May_.log
│   ├── main.py
│   ├── Network.py
│   ├── networks
│   │   ├── BIRD.py
│   │   └── __pycache__
│   │       └── BIRD.cpython-36.pyc
│   ├── nohup.out
│   ├── __pycache__
│   │   └── Network.cpython-36.pyc
│   ├── run
│   │   └── run_bird.sh
│   └── weights
│       └── PPDD
│           └── BIRD
│               ├── BIRD.data-00000-of-00001
│               ├── BIRD.index
│               ├── BIRD.meta
│               └── checkpoint
├── TensorFlow
│   └── layers
│       ├── BPTRUCell.py
│       ├── __init__.py
│       └── __pycache__
│           ├── BPTRUCell.cpython-36.pyc
│           └── __init__.cpython-36.pyc
└── utils
    ├── data_loader.py
    ├── __init__.py
    ├── __pycache__
    │   ├── data_loader.cpython-36.pyc
    │   ├── __init__.cpython-36.pyc
    │   └── utility.cpython-36.pyc
    └── utility.py
   
```

## Prepare for the Project

The path of this project in my server is: `/home/guoxiu.hgx/hgx/Research/BIRD`.

* Option 1:
  * `cd ~; mkdir hgx; cd hgx; mkdir Research; cd Research`
  * `git clone https://github.com/GuoxiuHe/BIRD.git`
  
* Option 2:
  * `cd your_own_path`
  * `git clone https://github.com/GuoxiuHe/BIRD.git`
  * modify the code in every script refer to your own path:
  
    ```
    curdir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(curdir))
    rootdir = '/'.join(curdir.split('/')[:4])
    PRO_NAME = 'BIRD'
    prodir = rootdir + '/Research/' + PRO_NAME
    sys.path.insert(0, prodir)
    ```

## Prepare for the Dataset
* Download Dataset from [This URL](https://drive.google.com/file/d/1uiUNp7DdPD_yYX8v8BjOFLJu0-CldnsQ/view?usp=sharing).
  * I am very grateful to Miss Sisi Gui for her effort to make another copy on Baidu Network Disk, which will be very friendly to Chinese researchers:
    ```
    link: https://pan.baidu.com/s/1n15POLjeG66uiU4KVJUtgw
    code：rmqn 
    ```
* Put the Dataset to the right place as the structure shows.
* Data Structure:

    ```
    {
    "user_id": "XXXXXX",
    "item_id": "XXXXXX",
    "title": "XXXXXX",
    "session": [["query1", ["product1", "product2", ...]],
                ["query2", ["product3", "product4", ...]],
                ...
               ]
    "label": "black_list"
    }
    ```

## How to Reproduce the Results

* Note that we have made a pre-trained BIRD model publicly available. 

* train a model: 
  * `CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase train --model_name bird --data_name ppdd --memory 0.45 &`

* local evaluation at session level:
  * `CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase evaluate_val --model_name bird --data_name ppdd --memory 0 &`
  * `CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase evaluate_test --model_name bird --data_name ppdd --memory 0 &`
  
* online testing at item level:
  * `CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase predict_items --model_name bird --data_name ppdd --memory 0 &`
  * `CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase predict_items2 --model_name bird --data_name ppdd --memory 0 &`
  
## Cite
If you use the codes or datasets, please cite the following paper:

```
@inproceedings{he2019finding,
  title={Finding Camouflaged Needle in a Haystack?: Pornographic Products Detection via Berrypicking Tree Model},
  author={He, Guoxiu and Kang, Yangyang and Gao, Zhe and Jiang, Zhuoren and Sun, Changlong and Liu, Xiaozhong and Lu, Wei and Zhang, Qiong and Si, Luo},
  booktitle={Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={365--374},
  year={2019},
  organization={ACM}
}
```