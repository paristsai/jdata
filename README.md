# JData 京東算法 - 高潛用戶購買意向預測


## Problem Description

這個競賽描述的是一個電商經典場景 - 精準行銷

這個用戶在網站上累積了這麼多歷史足跡，我們總想知道：

1. 用戶會消費嗎？
    - 如果不會的話，可以即時透過推播、行銷 EDM、簡訊甚至是限時優惠 Coupon 提醒他們回來。 
2. 若是會，他們會買什麼？
    - 如果已經知道用戶極有可能會購買某項產品，我們可以...
        - 透過成本較低的行銷方法促進消費
        - 提供相似產品競爭機會
        - 推薦利潤較高產品
        - 預測產品庫存


而這個賽事正是應用這個場景的最佳範例。

詳細的介紹可以參考[賽事官網](https://www.datafountain.cn/competitions/247/details/)

## Data

請至這個[連結](https://pan.baidu.com/s/1i4QC8lv)下載，並且放到 data/raw/ 資料夾中


## Evaluation

計分方式採用經典的 F1，意即需要兼顧 precision 和 recall。F1 又分為兩種，一種是 user 比重 0.4，另一種是 user-sku pair 比重 0.6，且只計算正樣本。詳情請參考 [1](https://www.datafountain.cn/competitions/247/details/data-evaluation) [2](https://www.datafountain.cn/competitions/247/details/faq)。

我們必須根據用戶從 2016-02-01 到 2016-04-16 的累積的行為資料，預測 2016-04-16 到 2016-04-20 這 5 天會購買什麼商品，但由於這個賽事已經關閉提交結果的功能，我們沒辦法知道預測的結果是否正確，只能將線下的資料切割成訓練集與測試集，將測試集當成正確答案。而且為了更加符合一週 7 天的情況，測試集的時長將拉到一週。

也就是說，我會將 2016-02-01 ~ 2016-04-09 的資料作為訓練集，2016-04-09 ~ 2016-04-16 實際購買的 user_id, sku_id pair 當成正確答案，透過優化特徵與模型，提高分數。

## Solution

version 1.0:
- model: only xgboost classifier
- choose the random under sampling ratio 0.01
- feature 1.0:
    - add the basic aggregation features (counts and unique counts) with window sizes
        - user
        - item
        - user-item
    - add the user item interaction time features
- results
    - single best model
        - 0.144194/0.085496 -> 0.108975
    - voting from the 3 resampling models
        - 0.132765/0.08477 -> 0.103968


## Training Requirements
Python 3.7
```
make create_environment
make data
```

--------
## Project Organization
<p><small> based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
