# 研究计划：气候变化对不同水文气候区径流影响的量化与归因

## 1. 研究题目

**气候变化对不同水文气候区径流影响的量化与归因——基于ERA5-Land的沙特阿拉伯、意大利与孟加拉国对比研究**

**Quantifying and Attributing Climate Change Impacts on Runoff across Different Hydroclimatic Regimes: A Comparative Study of Saudi Arabia, Italy, and Bangladesh Using ERA5-Land**

---

## 2. 研究背景与动机

气候变化通过改变降水模式、温度和蒸散发等过程，深刻影响全球水循环和区域径流产生机制。然而，气候变化对径流的影响在不同气候区存在本质差异：

- 在**极端干旱区**（如沙特阿拉伯），降水稀少且高度不稳定，径流主要由极端降水事件驱动，温度升高可能通过增强蒸散发进一步削减有效径流。
- 在**半干旱-亚湿润过渡区**（如意大利），地中海气候与温带气候并存，南北降水梯度显著，径流同时受水分供给与能量条件控制，是理解气候变化影响的关键过渡带。
- 在**极端湿润区**（如孟加拉国），降水充沛，径流受季风系统性控制，气候变化可能通过改变季风强度和降水季节分配影响径流。

三个国家沿全球干湿梯度分布，覆盖了从极端干旱到极端湿润的完整谱系。传统方法难以量化气候因素对径流变化的具体贡献，本研究引入XGBoost结合SHAP（SHapley Additive exPlanations）可解释性归因框架，量化各气候驱动因素对径流变化的贡献，并对比三种水文气候区的驱动机制差异。

---

## 3. 研究目标

1. 分析1950—2025年间三国径流及相关气候变量的时空变化趋势。
2. 构建水量平衡方程，诊断三国水量平衡各分量（P、ET、R、ΔS）的长期变化。
3. 利用XGBoost + SHAP方法，量化气候因素对径流变化的贡献，识别三国径流变化的主控因素及其沿干湿梯度的变化规律。

---

## 4. 科学问题与假设

### 科学问题

- **RQ1**：1950—2025年间，三国径流分别呈现怎样的时空变化趋势？
- **RQ2**：三国水量平衡各分量（P、ET、R、ΔS）如何随气候变化而改变？
- **RQ3**：哪些气候因素对三国径流变化的贡献最大？径流主控因素是否沿干湿梯度呈现系统性变化？

### 假设

- **H1**：沙特径流变化主要由降水驱动，意大利由降水和蒸散发相关因素协同控制，孟加拉国径流受季风降水绝对主导。
- **H2**：从干旱区到湿润区，降水对径流的SHAP贡献量级系统性增大，且年度模型表现优于月度模型。
- **H3**：意大利月度模型中土壤温度（季节代理变量）重要性高，年度模型中回归为降水主导，体现尺度效应。

---

## 5. 研究区域

### 5.1 沙特阿拉伯（极端干旱区）

- 范围：16°N—33°N，36°E—56°E
- 气候特征：极端干旱（柯本BWh/BWk），年均降水量 ~67 mm（ERA5-Land实测）
- 水量平衡：ET/P = 1.11（ERA5-Land存在已知高估，可能反映化石地下水蒸散发），R/P = 3.2%
- 径流特征：以洪泛产流（Wadi flash flood）为主，径流系数极低

### 5.2 意大利（半干旱至亚湿润过渡区）

- 范围：36°N—48°N，6°E—19°E
- 气候特征：南部地中海气候（Csa），北部温带（Cfa/Cfb），年均降水量 ~1047 mm
- 水量平衡：ET/P = 0.61，R/P = 41.2%
- 径流特征：季节性显著，存在南北梯度，高山区有融雪径流

### 5.3 孟加拉国（极端湿润区）

- 范围：20°N—27°N，88°E—93°E
- 气候特征：热带季风气候（Am），年均降水量 ~2316 mm
- 水量平衡：ET/P = 0.46，R/P = 56.2%
- 径流特征：恒河-布拉马普特拉河系，季风驱动，洪水频发

### 干湿梯度对比

| 特征 | 沙特阿拉伯 | 意大利 | 孟加拉国 |
|------|----------|--------|---------|
| 年降水量 | ~67 mm | ~1047 mm | ~2316 mm |
| ET/P | 1.11 | 0.61 | 0.46 |
| R/P（径流系数） | 3.2% | 41.2% | 56.2% |
| XGBoost月度 R² | 0.334 | 0.816 | 0.936 |
| XGBoost年度 R² | 0.559 | 0.970 | 0.984 |
| 径流主控因素 | 极端事件（月均不可预测） | 季节性能量-水分协同 | 降水绝对主导 |

---

## 6. 数据来源

### 6.1 气候数据

- **数据集**：ERA5-Land Monthly Aggregated（`ECMWF/ERA5_LAND/MONTHLY_AGGR`）
- **获取平台**：Google Earth Engine（GEE）
- **空间分辨率**：0.1°（约11 km）
- **时间范围**：1950年2月—2025年12月（共911个月）
- **提取脚本**：`gee/era5land_extraction.js`

### 6.2 地形数据

- **数据集**：MERIT DEM v1.0.3（Yamazaki et al., 2017）
- **GEE数据集ID**：`MERIT/DEM/v1_0_3`
- **空间分辨率**：3弧秒（~90 m）
- **用途**：研究区地图（fig01）地形填充
- **提取脚本**：`gee/merit_dem_extraction.js`
- **本地路径**：`data/DEM/Saudi_DEM.tif`、`Italy_DEM.tif`、`Bangladesh_DEM.tif`

### 6.3 变量清单

#### 水量平衡分量

| 变量 | GEE波段名 | 单位转换 |
|------|----------|---------|
| 总降水量（P） | total_precipitation_sum | m → mm（×1000） |
| 总蒸散发（ET） | total_evaporation_sum | m → mm（×−1000，取绝对值） |
| 地表径流 | surface_runoff_sum | m → mm |
| 地下径流 | sub_surface_runoff_sum | m → mm |
| 总径流（R） | surface + subsurface | mm |
| 土壤水储量（S，4层） | volumetric_soil_water_layer_1–4 | m³/m³ → mm（分层系数：70/210/720/1890） |

#### XGBoost特征变量（8个）

| 特征 | 物理意义 |
|------|---------|
| P_mm（当月降水） | 直接水分输入 |
| T_C（2m气温） | 能量/蒸散发驱动 |
| Td_C（露点温度） | 大气湿度 |
| Rn_sw（净短波辐射） | 太阳能量输入 |
| Rn_lw（净长波辐射） | 长波辐射收支 |
| Wind（风速） | √(u²+v²)合量 |
| Ts_C（土壤温度） | 土壤热状态 |
| dS（土壤水变化量） | 前期湿润条件/记忆效应 |

---

## 7. 研究方法

### 7.1 技术路线

```
ERA5-Land Monthly GeoTIFF栈（GEE导出，1950-02至2025-12，911个月）
        │
        ├── NetCDF转换（00_convert_to_netcdf.py）→ data/netcdf/（3个压缩NetCDF）
        │
        ├──① 国家空间均值时间序列
        │        → 水量平衡分析（P = ET + R + ΔS + ε）
        │          fig01–fig05
        │
        ├──② 年度GeoTIFF栈（年总量/年均值，76年）
        │        → 像素级Mann-Kendall趋势检验
        │          fig06–fig08
        │
        └──③ 像素级月度DataFrame（月度模型）
                 → XGBoost训练（train: year≤2004, test: year≥2005）
                 → SHAP归因（月度+年度双模型）
                   fig09–fig13b

MERIT DEM（GEE导出，90m分辨率，三国各一文件）→ fig01地形填充
```

### 7.2 水量平衡分析

水量平衡方程：**P = ET + R + ΔS + ε**

- 数据来源：国家空间均值时间序列（`utils_load.load_country_mean_timeseries()`）
- 分析内容：年尺度时间序列、水量平衡结构对比、年径流系数 RC = R/P
- 注：沙特ε > 0（ET + R > P），属于ERA5-Land在极端干旱区的已知偏差，在Discussion中说明

### 7.3 趋势分析

#### 区域平均趋势（fig06、fig07）

- **Mann-Kendall趋势检验** + **Sen's斜率估计**
- 对P、ET、R、T的年值，同时分析年尺度和季节尺度（DJF、MAM、JJA、SON）
- α = 0.05显著性阈值

#### 像素级空间趋势（fig08）

- 对P、R、T三个变量逐像素计算Sen's斜率和Kendall τ p值
- 显著像素（p<0.05）全不透明，非显著像素半透明
- 数据来源：`data/GEE_ERA5Land_Annual/`年度GeoTIFF栈

### 7.4 XGBoost + SHAP 归因分析

#### 月度模型

- **输入**：像素级月度DataFrame（~1583万行/沙特，~308万行/意大利，~126万行/孟加拉）
- **目标变量**：月总径流 R（mm/月）
- **训练/测试划分**：year ≤ 2004（训练，~73%）/ year ≥ 2005（测试，~27%）
- **超参数**：n_estimators=200, max_depth=5, lr=0.1, subsample=0.8, colsample_bytree=0.8
- **评估指标**：R²、RMSE、NSE、KGE
- **SHAP计算**：测试集随机抽取5000样本（TreeExplainer）

#### 年度模型（新增）

- **输入**：月度数据聚合为像素-年（通量变量取年总量，状态变量取年均值）
- **样本规模**：沙特 ~132万像素-年，意大利 ~26万，孟加拉 ~11万
- **目标变量**：年总径流 R（mm/year）
- **优势**：消除季节内噪声，R²显著提升（沙特0.334→0.559，意大利0.816→0.970，孟加拉0.936→0.984）

#### SHAP分析输出

| 图 | 内容 |
|----|------|
| fig09/09b | 月度/年度预测 vs 实际散点图（R²、RMSE、NSE、KGE标注） |
| fig10/10b | 月度（3年滚动均值）/年度 SHAP 时序演变图 |
| fig11/11b | 月度/年度 SHAP Beeswarm Summary Plot |
| fig12/12b | 月度/年度特征重要性三国对比条形图 |
| fig13/13b | 月度/年度降水-SHAP Dependence Plot（颜色=温度） |

---

## 8. 主要结果

### 8.1 水量平衡

| 国家 | P（mm/yr） | ET（mm/yr） | R（mm/yr） | RC |
|------|----------|-----------|-----------|-----|
| 沙特阿拉伯 | 67.1 | 72.2 | 2.5 | 3.2% |
| 意大利 | 1046.7 | 633.4 | 435.0 | 41.2% |
| 孟加拉国 | 2316.0 | 1052.0 | 1316.9 | 56.2% |

### 8.2 趋势分析主要发现

- **三国共同**：显著增温趋势（p < 0.001）
- **孟加拉国**：年降水量（−4.1 mm/yr, p<0.05）和径流（−3.7 mm/yr, p<0.05）均呈显著下降趋势
- **意大利**：ET显著增加（+0.88 mm/yr, p<0.001），径流无显著趋势
- **沙特**：降水和ET无显著趋势，径流微弱下降（p<0.01）
- **空间异质性**：意大利北部P/R增加趋势与南部减少趋势分异明显（fig08）

### 8.3 SHAP 归因主要发现

**月度模型（fig11、fig12）：**
- 沙特：露点温度>气温>降水（月均变量与月均径流关联弱，R²=0.334，SHAP结果物理解释需谨慎）
- 意大利：土壤温度主导（季节代理变量，|SHAP|≈56 mm/month）
- 孟加拉国：降水绝对主导（|SHAP|≈73 mm/month）

**年度模型（fig11b、fig12b）：**
- 三国均以降水为最重要特征
- 沙特 |SHAP| < 5 mm/yr；意大利 ~270 mm/yr；孟加拉国 ~500 mm/yr
- 意大利年度模型中土壤温度退居次位（季节信号消除后，降水的跨年际主导地位显现）
- 孟加拉国 Antecedent Soil Water 排第二（土壤饱和的正反馈机制）

**核心结论**：沿干湿梯度，径流的气候可预测性和降水主导力系统性增强；干旱区径流由随机极端事件决定，月均气候变量不足以预测；湿润区径流与年降水量几乎线性相关。

---

## 9. 图表清单（fig01–fig13b，共18张）

| 图号 | 内容 | 脚本 |
|------|------|------|
| fig01 | 研究区地图（MERIT DEM填充，底部高程colorbar） | 01_water_balance.py |
| fig02 | 多年均值P/ET/R空间分布（3行×3列） | 01_water_balance.py |
| fig03 | 年度水量平衡时间序列（P/ET/R，三国） | 01_water_balance.py |
| fig04 | 水量平衡结构柱状图（ET/P、R/P三国对比） | 01_water_balance.py |
| fig05 | 年径流系数（R/P）年际变化 | 01_water_balance.py |
| fig06 | 年际趋势时间序列（P/ET/R/T，三国）| 02_trend_analysis.py |
| fig07 | 趋势热图（Sen's斜率×季节，三国×4变量） | 02_trend_analysis.py |
| fig08 | 像素级Sen's斜率空间趋势图（P/R/T，3×3） | 02_trend_analysis.py |
| fig09 | 月度 XGBoost 预测 vs 实际散点图 | 03_xgboost_shap.py |
| fig09b | 年度 XGBoost 预测 vs 实际散点图 | 03_xgboost_shap.py |
| fig10 | 月度 SHAP 时序演变（3年滚动均值） | 03_xgboost_shap.py |
| fig10b | 年度 SHAP 时序演变（逐年，无平滑） | 03_xgboost_shap.py |
| fig11 | 月度 SHAP Beeswarm Summary Plot | 03_xgboost_shap.py |
| fig11b | 年度 SHAP Beeswarm Summary Plot | 03_xgboost_shap.py |
| fig12 | 月度特征重要性三国对比条形图 | 03_xgboost_shap.py |
| fig12b | 年度特征重要性三国对比条形图 | 03_xgboost_shap.py |
| fig13 | 月度降水-SHAP Dependence Plot（颜色=温度） | 03_xgboost_shap.py |
| fig13b | 年度降水-SHAP Dependence Plot | 03_xgboost_shap.py |

---

## 10. 报告结构（不超过10页）

| 章节 | 页数 | 内容要点 |
|------|------|---------|
| Title + Authors | 0.5页 | 题目、姓名、贡献声明 |
| Introduction | 1.5页 | 气候变化与水循环、干湿梯度差异、SHAP方法引入、研究目标 |
| Study Area | 0.5页 | fig01研究区地图、三国水文气候特征对比表 |
| Data and Methods | 2页 | ERA5-Land数据描述、GeoTIFF→NetCDF流程、水量平衡方法、MK趋势、XGBoost+SHAP（双模型设计） |
| Results | 3—4页 | fig02–05（水量平衡）、fig06–08（趋势）、fig09/09b（模型性能）、fig11b/12b（年度SHAP）、fig13b（依赖图） |
| Discussion | 1页 | 干湿梯度机制变化、月度 vs 年度模型尺度效应、ERA5-Land局限性（沙特ET>P） |
| Conclusion | 0.5页 | 主要发现 |
| References | 1页 | |

---

## 11. 核心参考文献

1. Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*.
2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
3. Yamazaki, D., et al. (2017). A high-accuracy map of global terrain elevations. *Geophysical Research Letters*. (MERIT DEM)
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
5. Gupta, H. V., et al. (2009). Decomposition of the mean squared error and NSE. *Journal of Hydrology*. (KGE)
6. Budyko, M. I. (1974). *Climate and Life*. Academic Press.
7. Nearing, G. S., et al. (2021). What role does hydrological science play in the age of machine learning? *Water Resources Research*.

---

## 12. 工具与环境

| 工具 | 用途 |
|------|------|
| Google Earth Engine（JavaScript） | ERA5-Land + MERIT DEM 提取、预处理、GeoTIFF导出 |
| Python xarray / rioxarray / rasterio | GeoTIFF→NetCDF转换、像素级数据管理、DEM读取 |
| Python pandas / numpy | 时间序列处理与水量平衡计算 |
| Python pymannkendall | Mann-Kendall趋势检验 |
| Python scipy | 像素级Sen's斜率（theilslopes）和Kendall τ（kendalltau） |
| Python xgboost | 机器学习模型训练与评估 |
| Python shap | SHAP可解释性分析（TreeExplainer） |
| Python matplotlib / cartopy | 可视化与地图制作 |
| Conda环境 | `D:/anaconda3/envs/hydroclimate/` |
