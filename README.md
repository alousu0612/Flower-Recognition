# Flower-Recognition
Use Deep Learning Structure to Recognize Flower Type

花朵辨識，選定的範圍內共有五種不同種類的花 : 雛菊(daisy)、蒲公英(dandellion)、玫瑰(rose)、向日葵(sunflower)、鬱金香(tulip)，請以同學使用訓練資料當中的照片，並應用在深度學習階段所學到的內容，來辨識照片中的是哪種花。

本測驗的目的，在於讓同學練習並熟悉影像辨識的做法，實際操作後半部課程的內容。尤其是一般 CNN模型與 Pre-training model 的差距，也希望同學能透過這次測驗體驗到。

特徵說明
圖形辨識的特徵就是圖檔本身，因此訓練特徵就是圖片本身，不另做說明。而作答的 id 就是檔名，同學可以詳閱 "Data" 分頁的說明以及 sample_submission.csv 的內容。

比較不同的是預測的輸出值，請同學特別注意 : 以數字 0 / 1 / 2 / 3 / 4 輸出你的要提交預測類別，而不是以花朵名稱輸出 ( 建議以 Python Dictionary 轉換，或輸出時直接是類別碼，例如 : flower_mapping = {'daisy':0, 'dandelion':1, 'rose':2, 'sunflower':3, 'tulip':4} )
