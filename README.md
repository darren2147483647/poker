# poker

這筆資料集紀錄2024年WSOP Paradise主賽事最後一天的牌局，這場比賽是25,000美元買入的超級主賽，保證獎池為50,000,000美元。最終中國選手Yinan Zhou奪得冠軍並獲得大約6,000,000美元的獎金，也是第一位獲得WSOP主賽事冠軍的亞州選手。

資料以影片時間軸排序，每一局以一份.phh格式(poker hand history format)文件紀錄，包含賽事類型、盲注級別、所有玩家的人名、起始籌碼、最終籌碼、每一手牌、以及每一筆下注、跟注、和棄牌動作。

程式部分，提供分析資料的AI範例，但建議使用資料集的人自己寫更合適的程式碼，一場牌局有很多資訊是可以用來分析的。

建議使用pokerkit工具解析phh文件，雖然分析用的範例程式不會用到。

受University of Toronto Computer Poker Student Research Group啟發。

未來會補齊剩餘的牌局，或新增更多撲克賽事的紀錄。

## dataset

*   hand_record/

    WSOP Paradise Event #9 第五天的牌局內容，以時間軸排序，全部以.phh格式紀錄，一份.phh文件只記錄一局。
    我觀看WSOP Youtube頻道的影片直播紀錄並手動標注，目前記錄到影片01:10:40，共16手牌。
    不知是否為賽制原因，第16手沒有小盲玩家也沒人支付小盲，因此第16結束後籌碼分布與程式結果不同，實際如下:
    [214500000, 52500000, 237000000, 86500000, 160000000, 90000000, 100000000]
    Mustapha Kanit有214500000，做在D位而非SB位，也不支付2M的小盲注。Marcelo Aziz也少贏2M。

*   cprg_hands_fixed.json

    由IRC撲克遊戲資料集擷取的所有牌局紀錄(僅限No-limit無限注模式)。IRC撲克遊戲伺服器是較早出現的線上撲克遊戲，曾聚集許多撲克遊戲愛好者，現在已經停運。
    這筆資料來源自Michael Maurer's IRC Poker Database，利用Github用戶allenfrostline與Miami Data Science Meetup projects提供的程式碼收集，但資料儲存的方式不夠完整(缺少大多數玩家的底牌與每一筆raise數目)，還存在一些不被撲克遊戲定義的行為(像是被踢出房間)，因此無法轉換為PHH格式或其他可讀性高的形式，作為附錄放在一邊，並提供來源與相關的文件(IRC dataset)。

## code

*   better_poker_recorder.py

    紀錄.phh文件的程式，先讀取上一局.phh文件的結果，沿用設定建立新牌局，就可以開始記錄了，分為自動與手動兩種模式，自動模式下讀取rec.txt文件並記錄，將我習慣的簡記方式轉換成標準格式，若rec.txt文件為空則為手動模式，需要逐行輸入牌局上的每一手動作，嚴格按照標準phh文件的action模式。
*   label_extract.py

    為了訓練AI，從.phh文件提取重要資訊，範例程式需要import此程式，但不建議用在其他用途，同需求者建議自己寫。
*   kmean.py

    範例程式，非監督，Kmean演算法
*   DBSCAN.py

    範例程式，非監督，DBSCAN演算法
*   regression.py

    範例程式，監督，linear regression模型，DLP
*   adaboost.py

    範例程式，監督，Adaboost演算法

## third party

*   licence/

    PokerKit許可證

## requirement

PokerKit
Torch
Sklearn
numpy
matplotlib

## 相關連結

*   PHH格式標準 (document連結)
https://arxiv.org/html/2312.11753v2/#S6

*   PokerKit (Github連結)
https://github.com/uoftcprg/pokerkit?tab=readme-ov-file

*   2024 World Series of Poker Paradise Event #9: $25,000 Super Main Event | Day 5 (FINAL TABLE) (YT連結)
https://www.youtube.com/watch?v=-bNvm2hx3MQ

*   University of Toronto Computer Poker Student Research Group phh資料集 (github連結)
https://github.com/uoftcprg/phh-dataset/tree/main

## IRC dataset

*   主來源  IRC dataset
https://poker.cs.ualberta.ca/irc_poker_database.html

*   解析程式1  由Github用戶allenfrostline提供
https://github.com/allenfrostline/PokerHandsDataset

*   解析程式2  由Github用戶dksmith01提供
https://github.com/dksmith01/MSDM/blob/987836595c73423b89f83b29747956129bec16c2/.ipynb_checkpoints/MDSM%20Project%201%20Poker%20Python%20Wrangling%20Code-checkpoint.ipynb