## data/README.md

```markdown
# 数据设置说明

此文件夹包含 ELECTRIcity 项目的处理后数据和原始数据。

## 目录结构
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

## 设置说明

1. 从各自的来源下载原始数据集：
   - [REDD](http://redd.csail.mit.edu/)
   - [UK-DALE](https://jack-kelly.com/data/)
   - [REFIT](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)

2. 将下载的数据放在适当的 `raw` 目录中。

3. 对于 REFIT，将 .csv 文件放在 `Refit/Data` 文件夹中，将标签文件放在 `Refit/Labels` 文件夹中。

## 格式要求

### REDD 数据集
- 每个房屋应该有自己的文件夹，命名为 `house_X`，其中 X 是房屋编号
- 每个房屋文件夹内应该有：
  - `labels.dat`：包含设备标签
  - `channel_1.dat`、`channel_2.dat` 等：包含功率测量值

### UK-DALE 数据集
- 与 REDD 类似，每个房屋应该有自己的文件夹，命名为 `house_X`
- 每个房屋文件夹应包含：
  - `labels.dat`：包含设备标签
  - `channel_1.dat`（总功率），以及其他用于单个设备的通道

### REFIT 数据集
- CSV 文件应命名为 `HouseX.csv`，其中 X 是房屋编号
- 标签文件应命名为 `HouseX.txt`，其中 X 是房屋编号
- 将 CSV 文件放在 `Data` 文件夹中，将标签文件放在 `Labels` 文件夹中

## 自动处理

当您运行训练脚本时，代码将自动将原始数据处理为存储在 `processed` 目录中的 CSV 文件。这种处理对于每种数据集配置只需要进行一次。