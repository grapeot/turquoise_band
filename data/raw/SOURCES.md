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

## 太阳谱 — `sao2010_solref.dat`（真实测量 AM0 太阳谱，已下载）
- 来源：SAO2010 / Chance & Kurucz 2010 高分辨率太阳参考谱，Harvard-Smithsonian CfA Atmospheric Spectroscopy 组发布
- 直链：http://www.cfa.harvard.edu/atmosphere/links/sao2010.solref.converted （下载时 HTTP 200，text/plain）
- 引用：Chance, K. and Kurucz, R.L., "An improved high-resolution solar reference spectrum for Earth's atmosphere measurements in the ultraviolet, visible, and near infrared", J. Quant. Spectrosc. Radiat. Transfer 111, 1289-1295 (2010)
- 这是 AM0（大气外）太阳谱，0.01nm 高分辨率，能分辨 Fraunhofer 吸收线——比 5772K 黑体更接近真实蓝端衰减。

### 格式（可直接 numpy.loadtxt 解析）
- 5 行文本头（4 行列说明 + 1 行空行），**加载时 `skiprows=5`**，空白分隔。
- 4 列：
  - 列1 = **真空波长 (nm)**（注意是真空波长，不是空气波长；与 o3_serdyuchenko 一致，无需互转）
  - 列2 = 光子辐照度 photons·s⁻¹·cm⁻²·nm⁻¹
  - 列3 = **辐照度 W·m⁻²·nm⁻¹**（直接可用，渲染优先用这一列，无需任何转换）
  - 列4 = 辐照度 W·m⁻²·cm（按波数，一般用不到）
- 数据行 80093 行，波长范围 **200.07 – 1000.99 nm**，步长恒定 0.01nm。完整覆盖可见光 380–780nm。
- 加载示例：`d = np.loadtxt('sao2010_solref.dat', skiprows=5); wl, irr = d[:,0], d[:,2]`

### 单位换算（仅当想从列2自行推导列3时；已验证两列自洽，相对误差 <5e-6）
- W/m²/nm = photons/cm²/s/nm × (h·c/λ) × 1e4
  - h = 6.62607015e-34 J·s，c = 2.99792458e8 m/s，λ 用**米**（nm×1e-9）
  - ×1e4 是 cm⁻² → m⁻² 的面积换算（1 m² = 1e4 cm²）

### 与 5772K 黑体的差异（550nm 归一化后的相对比值，确认蓝端偏低）
- 400nm: 0.92（真实太阳显著偏低，Fraunhofer 线密集 + Ca H&K/Balmer 吸收）
- 450nm: 0.99，550nm: 1.00（基准），650nm: 0.93，700nm: 0.93
- 这正是绿松石带颜色可能受影响的物理来源，替换黑体后应对比色相曲线确认无回归。

### 关于 ASTM E-490（未下载，记录原因）
- 原计划首选 ASTM E-490 AM0（W/m²/nm，~1nm），但其唯一权威宿主 NREL/rredc.nrel.gov 在当前环境 DNS 无法解析（HTTP 000），所有镜像（UMaine misclab、Sandia、pvlib）的数据链接都回指 NREL 或为 404。
- pvlib 仅内置 ASTM **G173**（AM1.5 地面谱，含大气吸收），不是 AM0，不适用。
- SAO2010 是更优替代：同为 AM0，分辨率高 100 倍，且自带 W/m²/nm 列。E-490 如日后需要，可手动从 NREL 网页下 e490_00a_amo.xls 补入。

## 地球纹理 — `earth_texture/`
- **NASA Blue Marble: Next Generation w/ Topography and Bathymetry**（2004年7月合成），公有领域。
  Visible Earth 页面：https://visibleearth.nasa.gov/images/73909/december-blue-marble-next-generation-w-topography-and-bathymetry
  （该记录同时托管 1–12 月各月合成；July=200407）
- `earth_bluemarble_topobathy_200407_5400x2700.jpg`：白天面真彩 albedo（海洋蓝 + 大陆 + 地形/水深晕渲），5400×2700，baseline JPEG，RGB 三通道，plate carrée（等距柱状），经度 0° 居中，纬度 ±90°，全球无缝。
  - 直链：https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg （下载时 HTTP 200，image/jpeg，2.2MB；NASA 站点需带浏览器 User-Agent，否则部分路径会被拒）
- 验证：mode=RGB，size=(5400,2700)，W/H=2.0（确认 plate carrée）；mean(maxch−minch)=40.7（远大于 0，确认彩色非灰度）；70% 像素蓝色通道占优（海洋）。
- 用途：贴到「从月球看地球」的圆盘上。月食/日食时地球背光面（夜面）朝向月球，渲染时把这张白天纹理整体压暗作夜面剪影；保留海陆轮廓用于地球盘的明暗与边缘大气环定位。
- 备选（未下载）：同记录有 `...3x21600x10800.jpg`（21600×10800 全分辨率，约 80MB）。

### 夜面灯光 — `earth_blackmarble_2016_3600x1800.jpg`（NASA Black Marble 2016 / Earth at Night，已下载）
- 来源：**NASA Earth Observatory / Earth at Night "Black Marble" 2016 合成**（VIIRS DNB 城市灯光），公有领域。
  Visible Earth 记录 144898：https://visibleearth.nasa.gov/images/144898/earth-at-night-black-marble-2016-color-maps
  Earth Observatory Flat Maps 页面：https://science.nasa.gov/earth/earth-observatory/earth-at-night/maps
- `earth_blackmarble_2016_3600x1800.jpg`：地球夜面真彩灯光图，3600×1800，baseline JPEG，RGB 三通道，plate carrée（等距柱状），经度 0° 居中，纬度 ±90°，全球无缝。这是 0.1°/px（01deg）档；典型「2k–5k」尺寸，体积小（761KB）。
  - 直链：https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/144000/144898/BlackMarble_2016_01deg.jpg （下载时 HTTP 200，image/jpeg，约 780KB；NASA assets 站点需带浏览器 User-Agent）
  - 更高清备选（未下载）：同记录 `BlackMarble_2016_3km.jpg`（3km/px ≈ 13500×6750，8.1MB）和 `..._geo.tif`（带地理参考的 GeoTIFF）；如需更密的城市灯光细节可换用，直链把 `01deg` 换成 `3km`。
- 验证：mode=RGB，size=(3600,1800)，W/H=2.0（确认 plate carrée 等距柱状）；全局平均亮度仅 19/255，73.6% 像素 lum<20（确认大部分为夜面黑底）；城市点抽样 Tokyo lum≈217、NYC lum≈201（明亮城市灯光），Sahara≈43（地表底图微亮），南太平洋≈8（深海漆黑）；经度映射命中（东京 lon139 在右半、纽约 lon-74 在左半，确认 0° 居中）。
- 用途：日食/月食时朝月球的是地球背光夜面，城市灯光可见。直接贴到「从月球看地球」圆盘的夜面，渲染城市灯光亮点；与 Blue Marble 白天纹理配合（白天纹理压暗作海陆轮廓 + 大气环，灯光图叠加发光层）。
