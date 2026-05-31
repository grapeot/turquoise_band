# 原始数据来源

## 臭氧吸收截面 — `o3_serdyuchenko_2014.dat`
- 来源：IUP Bremen MolSpec Lab，Serdyuchenko/Gorshelev/Weber/Burrows 2014（社区金标准，ACSO 推荐）
- Zenodo DOI: 10.5281/zenodo.5793207
- 格式：45 行文本头 + 88668 行数据，12 列。列1=真空波长(nm)，列2-12=截面(cm²/分子)，对应 11 个温度：293,283,273,263,253,243,233,223,213,203,193 K
- 波长网格：0.01nm 步长，213-1100nm。覆盖整个 Chappuis 带。
- 引用：Gorshelev et al. AMT 7,609-624 (2014); Serdyuchenko et al. AMT 7,625-636 (2014)
- 用法：平流层取 ~233K 列（列索引 7，0-based: 波长列0，233K 是第 7 列）

## 大气廓线 — `afgl_us_standard.csv`
- 来源：AFGL Atmospheric Constituent Profiles 1986，US Standard 模型（model 6 / table 1f），rayference/afgl1986 镜像
- 格式：CSV，51 行（0-120km），列：z(km), p(hPa), t(K), n(空气数密度 cm⁻³), H2O,O3,N2O,CO,CH4（均 ppmv VMR）
- 臭氧数密度 = O3[ppmv]·1e-6·n[cm⁻³]
- 引用：Anderson et al. AFGL-TR-86-0110 (1986); COESA US Std Atm 1976

## CIE CMF — `cie_xyz_1931_1nm.csv`
- 来源：CVRL (UCL)，CIE 1931 2° 标准观察者，1nm
- 格式：无表头 CSV，列：波长(nm), x̄, ȳ, z̄。覆盖 360-830nm
- 注：项目代码实际用 colour-science 库内置的等价 CMF；此文件留作独立校验

## 月面纹理 — `moon_texture/`
- **NASA CGI Moon Kit (SVS #4720)**，公有领域。页面 https://svs.gsfc.nasa.gov/4720
  基础路径 https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/
- `nasa_moon_color_lroc_4k_16bit.tif`：LROC WAC 彩色 albedo，4096×2048，16bit RGB，plate carrée，0°经度居中，近地面朝向观察者。渲染主力纹理。
- `nasa_moon_displacement_lola_ldem16_float_km.tif`：LOLA 高程图（位移），5760×2880 float32，单位 km，相对 1737.4km。供将来 3D 立体/法线渲染。
- 备选高清：color 8k/16k、displacement 64px/deg（URL 见 working.md，未下载）
- 回退：`moon-map-from-the-clementine-mission.png`（Clementine 1024×512 灰度）

## 太阳谱 — 待下载（可选 L1 升级）
- 计划：ASTM E-490 AM0 (NREL, W/m²/nm) 或 SAO2010 高分辨率
- 当前 v1 用 5772K 黑体近似（science review 确认对色相趋势足够）
- NREL: https://www.nrel.gov/grid/data/assets/data/e490_00a_amo.xls
