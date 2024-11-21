# 人工智能导论2024-作业3: Aliens游戏

在当前目录运行 `pip install -r requirements.txt` 安装所需代码库。

安装成功后，可运行 `python play.py` 用键盘方向键玩游戏并存储数据到 `logs/`。可通过修改 `level` 变量为 0~4 设置不同关卡。

存储数据后，修改 `learn.py` 中 `data_list` 变量为存储数据对应的路径，运行 `learn.py` 训练监督学习模型。运行结果将存储在 `logs` 目录下。

修改 `extract_features` 函数完成作业。