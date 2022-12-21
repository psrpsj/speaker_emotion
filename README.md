# speaker_emotion
DACON 월간 데이콘 발화자의 감정인식 AI 경진대회
## 1. Project Overview
  - 목표
    - 주어진 대화에서 발화자의 감정을 파악할 수 있는 AI모델 제작.
  - 모델
    - [tae898/emoberta-base](https://github.com/tae898/erc) fine-tuned model.
  - Data
    - 관광지 정보 (train: 9988개, test: 2609개).

## 2. Code Structure
``` text
├── data (not in repo)
|   ├── train.csv
|   ├── train_backtrans.csv
|   ├── train_upsample.csv
|   └── test.csv  
├── argument.py
├── backtranslation.ipynb
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── train.py
├── trainer.py
└── utils.py
```

## 3. Detail 
  - Preprocess 
    - nltk corpus의 stopword library를 활용하여 stopword 제거 전처리 수행.
  - Augmentation
    - 네이버 Papago를 이용한 데이터 수가 적은(200개 이하) label을 영어와 일어로 번역 후 한글로 재번역하는 Backtranslation을 이용한 augmentation 진행.
    - 갯수가 적은 label 데이터를 upsampling을 통한 augmentation 진행.
  - Model
    - 전처리 된 데이터를 tae898/emoberta-base에 fine-tunning 과정을 거침.
    - K-Fold(5 Fold)의 교차 검증과정을 통해 보다 정교한 모델 제작.
    - Focal loss를 사용, Imbalance한 데이터 분포에 보다 정교한 loss를 사용하여 모델 성능 향상.
    - Batch size: 16 / Epoch : 5
  - Inference
    - 각 Fold model의 inference 과정을 거친후 나온 softmax 확률을 soft-voting 과정을 거쳐 최종추론 함.
  - 최종성적
    - 대회기간 중 성적
      - Public Score: 0.4080
      - Private Score: 0.38559
    - 최종성적
      - Public Score: 0.4194
      - Private Score: 0.4098