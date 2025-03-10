NILM_Dataloader.py 应该放在 utils 目录下，它是一个工具类文件，用于创建 PyTorch 的 DataLoader 对象。

markdownCopy# ELECTRIcity

ELECTRIcity：一个高效的非侵入式负载监测变压器模型的 PyTorch 实现。如有任何问题，欢迎联系：stasykiotis@mail.ntua.gr

## 数据

您可以从以下链接下载 CSV 数据集：[REDD](http://redd.csail.mit.edu/)、[UK-DALE](https://jack-kelly.com/data/) 和 [Refit](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)

对于 Refit，我们使用了稍微不同的文件夹结构。我们创建了在数据处理过程中需要的带有列标签的 .txt 文件。请将 .csv 文件放在 Data 文件夹中以使代码正常工作。

data 文件夹的结构应为：
.
├── data
├── processed/         # 处理后的CSV文件（自动生成）
└── raw/               # 原始数据集
├── UK_Dale
│     ├── House_1
│     │    ├── .
│     │    └── .
│     └── House_2
│           .
│           .
├── REDD
│     ├── House_1
│     │    ├── .
│     │    └── .
│     └── House_2
│          ├── .
│          └── .
└── Refit
├── Data
│  House2.csv
│  House3.csv
│  House5.csv
│  House16.csv
└── Labels
House2.txt
House3.txt
House5.txt
House16.txt

ELECTRIcity/
├── data/
│   ├── processed/        # 将包含处理后的CSV文件（自动生成）
│   │   ├── redd_lf/
│   │   ├── uk_dale/
│   │   └── refit/
│   ├── raw/              # 原始数据
│   │   ├── REDD/         # REDD数据集
│   │   ├── UK_Dale/      # UK-DALE数据集
│   │   └── Refit/        # Refit数据集
│   │       ├── Data/
│   │       └── Labels/
│   └── README.md         # 数据使用说明
├── models/
│   ├── model_helpers.py  # 模型的辅助组件
│   └── electricity.py    # 主模型实现（包含 ELECTRICITY 类）
├── utils/
│   ├── config.py         # 配置设置和参数
│   ├── data_processor.py # 合并的数据处理功能
│   ├── metrics.py        # 评估指标
│   ├── dataset.py        # 数据集类
│   └── NILM_Dataloader.py # PyTorch数据加载器
├── main.py               # 主执行脚本
├── trainer.py            # 训练功能
└── README.md             # 项目文档
## 运行方法

本项目提供了使用 ELECTRIcity 训练模型的端到端流程。

运行代码所需的包可以在 electricity.yml 中找到。模型训练和测试可以通过运行 main.py 文件来完成。

```bash
python main.py
首先，config.py 提供了流程中所需的所有超参数。然后，脚本会根据用户在 config.py 中的选择（参数 dataset_code），创建 UK_Dale、Refit 或 Redd 的数据集处理器。trainer.py 包含执行模型训练和测试所需的所有函数。
模型训练和测试后，以下结果会被导出到 'results/dataset_code/appliance_name/' 目录：

best_acc_model.pth 包含导出的模型权重
results.pkl 包含训练期间记录的各种指标
test_result.json 包含测试期间的真实标签和模型预测结果

性能
我们使用 config.py 中可以找到的超参数，对每个数据集中的每个设备训练模型 100 个 epoch。
UK_Dale
<img src=results_uk_dale.png width=1000>
REDD
<img src=results_redd.png width=1000>
Refit
<img src=results_refit.png width=1000>

文件描述和运行说明
核心文件：

main.py：运行模型的入口点。处理从数据处理到训练和评估的整个管道。
models/electricity.py：包含 ELECTRICITY 模型实现，这是一种基于变压器的架构，用于非侵入式负载监控 （NILM）。
models/model_helpers.py： transformer 模型的辅助组件，包括注意力机制、位置编码等。
utils/config.py：配置设置和超参数，以及用于重现性的随机种子设置。
utils/data_processor.py：处理 REDD、UK-DALE 和 REFIT 数据集的加载和预处理。将处理后的数据输出为 CSV 文件，以避免重新处理。
utils/dataset.py：包含用于正常训练和带掩码的预训练的 PyTorch 数据集类。
utils/NILM_Dataloader.py：创建 PyTorch DataLoader 对象以实现高效的批处理。
utils/metrics.py：回归（能源预测）和分类（设备状态检测）的评估指标。
trainer.py：处理训练循环、验证和测试过程。

如何运行：

设置数据：

将原始数据集放在data/raw/
对于 REFIT，请确保您有单独的 Data 和 Labels 文件夹


配置：

如果需要，在 中修改参数config.py
重要参数包括：

dataset_code：在“redd_lf”、“uk_dale”或“改装”之间进行选择
appliance_names：要监控的设备列表
house_indicies： 用于训练的房屋清单




运行训练和测试：这将：bashCopypython main.py

处理数据（如果尚未处理）
训练模型（如果启用，则使用预训练）
在保留的测试屋上测试模型
保存结果和指标


查看结果：

模型权重将保存到results/{dataset_code}/{appliance_name}/best_acc_model.pth
指标和预测将保存到：

results/{dataset_code}/{appliance_name}/results.pkl
results/{dataset_code}/{appliance_name}/test_result.json





重要说明：

数据路径：

如果数据位置不同，则需要调整 中的路径config.py
默认路径假定数据放置在data/raw/{dataset_name}


重现性：

默认情况下，随机种子固定为 42 以实现可重复性
使用相同的种子运行相同的命令以获得相同的结果


GPU 使用情况：

如果可用，该代码将自动使用 CUDA
您可以通过在参数中设置来强制 CPU 使用率--device cpu


培训流程：

训练过程包括可选的预训练（掩码预测）
该模型可预测设备功耗和开/关状态
最佳模型是根据准确性、F1 分数和误差指标的组合保存的