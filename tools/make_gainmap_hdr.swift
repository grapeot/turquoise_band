// make_gainmap_hdr.swift —— Mac CLI：从 SDR base + 线性 HDR 生成带 gain map 的 HEIC
//
// 目标：下到 iPhone Photos 能显示 HDR（亮部超亮）。复用 YabiVision 的 ImageIO 写入管线，
// 但 gain map 从真实 HDR/SDR 比值**精确计算**（不是 YabiVision 那种从 SDR 猜）。
//
// 用法：
//   swift make_gainmap_hdr.swift <sdr.png> <hdr_linear.tif> <reference.HEIC> <out.heic>
// 编译版：
//   swiftc make_gainmap_hdr.swift -o make_gainmap -framework ImageIO -framework CoreGraphics -framework CoreImage
//
// gain map 原理（ISO 21496-1 / Apple Adaptive HDR）：
//   文件 = SDR base 图 + 单通道 gain map + 元数据。
//   gain[px] = log2(HDR_linear / SDR_linear) 归一化到 [0,255]。
//   Photos 读 gain map 把亮部推到 HDR 亮度；不支持的设备只看 SDR，向后兼容。

import Foundation
import ImageIO
import CoreGraphics
import CoreImage
import UniformTypeIdentifiers
import Accelerate

func fail(_ msg: String) -> Never { FileHandle.standardError.write((msg + "\n").data(using: .utf8)!); exit(1) }

let args = CommandLine.arguments
guard args.count == 5 else {
    fail("用法: swift make_gainmap_hdr.swift <sdr.png> <hdr_linear.tif> <reference.HEIC> <out.heic>")
}
let sdrURL = URL(fileURLWithPath: args[1])
let hdrURL = URL(fileURLWithPath: args[2])
let refURL = URL(fileURLWithPath: args[3])
let outURL = URL(fileURLWithPath: args[4])

// ---- 1. 读 SDR base（PNG, 8bit sRGB）为 CGImage ----
guard let sdrSrc = CGImageSourceCreateWithURL(sdrURL as CFURL, nil),
      let sdrImg = CGImageSourceCreateImageAtIndex(sdrSrc, 0, nil) else {
    fail("无法读取 SDR: \(sdrURL.path)")
}
let W = sdrImg.width, H = sdrImg.height
print("SDR base: \(W)x\(H)")

// 取 SDR 像素到线性（反 sRGB gamma），逐像素 RGBA8
func loadSDRLinear(_ img: CGImage) -> [Float] {
    var buf = [UInt8](repeating: 0, count: W * H * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(data: &buf, width: W, height: H, bitsPerComponent: 8,
                        bytesPerRow: W * 4, space: cs,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
    // sRGB → linear 的 luminance（用于 gain）
    var lin = [Float](repeating: 0, count: W * H)
    for i in 0..<(W * H) {
        let r = Float(buf[i*4+0]) / 255, g = Float(buf[i*4+1]) / 255, b = Float(buf[i*4+2]) / 255
        func s2l(_ c: Float) -> Float { c <= 0.04045 ? c/12.92 : pow((c+0.055)/1.055, 2.4) }
        lin[i] = 0.2126*s2l(r) + 0.7152*s2l(g) + 0.0722*s2l(b)
    }
    return lin
}
let sdrLin = loadSDRLinear(sdrImg)

// ---- 2. 读 HDR 线性 TIFF（float32 RGB, 含 >1）取 luminance ----
guard let hdrSrc = CGImageSourceCreateWithURL(hdrURL as CFURL, nil),
      let hdrImg = CGImageSourceCreateImageAtIndex(hdrURL == hdrURL ? hdrSrc : hdrSrc, 0, nil) else {
    fail("无法读取 HDR TIFF: \(hdrURL.path)")
}
guard hdrImg.width == W && hdrImg.height == H else {
    fail("HDR 与 SDR 尺寸不一致: \(hdrImg.width)x\(hdrImg.height) vs \(W)x\(H)")
}
// 用 CIImage 读 float TIFF，渲染到 float buffer
let ciHDR = CIImage(contentsOf: hdrURL)!
let ciCtx = CIContext(options: [.workingColorSpace: NSNull()])  // 不做色彩管理，保线性
var hdrBuf = [Float](repeating: 0, count: W * H * 4)
let fmt = CIFormat.RGBAf
let lcs = CGColorSpaceCreateDeviceRGB()
hdrBuf.withUnsafeMutableBytes { p in
    ciCtx.render(ciHDR, toBitmap: p.baseAddress!, rowBytes: W * 16,
                 bounds: CGRect(x: 0, y: 0, width: W, height: H), format: fmt, colorSpace: lcs)
}
var hdrLin = [Float](repeating: 0, count: W * H)
for i in 0..<(W * H) {
    hdrLin[i] = 0.2126*hdrBuf[i*4+0] + 0.7152*hdrBuf[i*4+1] + 0.0722*hdrBuf[i*4+2]
}
let hdrPeak = hdrLin.max() ?? 1
print("HDR 线性 luminance 峰值: \(hdrPeak)")

// ---- 3. 逐像素精确 gain map：gain = HDR/SDR，log2 编码到 8bit ----
// headroom = log2(max gain)；ISO 21496-1 的 gain map 存储归一化 log gain。
var ratios = [Float](repeating: 1, count: W * H)
for i in 0..<(W * H) {
    let s = max(sdrLin[i], 1e-4)
    let r = max(hdrLin[i] / s, 1.0)        // gain >= 1（HDR 不暗于 SDR）
    ratios[i] = r
}
let maxRatio = ratios.max() ?? 1
let logMax = max(log2(maxRatio), 0.1)      // headroom (stops)
print("最大 gain: \(maxRatio)x  (\(logMax) stops headroom)")

var gain8 = [UInt8](repeating: 0, count: W * H)
for i in 0..<(W * H) {
    let g = log2(ratios[i]) / logMax        // 归一化 [0,1]
    gain8[i] = UInt8(max(0, min(255, g * 255 + 0.5)))
}
let gainData = Data(gain8)

// ---- 4. 从 reference.HEIC 取 gain map 元数据 + DataDescription ----
guard let refSrc = CGImageSourceCreateWithURL(refURL as CFURL, nil),
      let refAux = CGImageSourceCopyAuxiliaryDataInfoAtIndex(refSrc, 0, kCGImageAuxiliaryDataTypeHDRGainMap) as? [String: Any] else {
    fail("无法从 reference.HEIC 取 gain map 元数据: \(refURL.path)")
}
let gainMeta = refAux[kCGImageAuxiliaryDataInfoMetadata as String] as! CGImageMetadata
var gainDesc = refAux[kCGImageAuxiliaryDataInfoDataDescription as String] as! [String: Any]
gainDesc["Width"] = W
gainDesc["Height"] = H
gainDesc["BytesPerRow"] = W
gainDesc["PixelFormat"] = kCVPixelFormatType_OneComponent8

// ---- 5. 写 HEIC：SDR base + Apple HDR 标记 + gain map 辅助数据 ----
var props = CGImageSourceCopyPropertiesAtIndex(sdrSrc, 0, nil) as? [String: Any] ?? [:]
var makerApple = props[kCGImagePropertyMakerAppleDictionary as String] as? [String: Any] ?? [:]
makerApple["33"] = 0.8       // Apple Adaptive HDR 标记（来自 YabiVision 验证）
makerApple["48"] = 0.0
props[kCGImagePropertyMakerAppleDictionary as String] = makerApple
props[kCGImagePropertyExifCustomRendered as String] = 3

let auxInfo: [String: Any] = [
    kCGImageAuxiliaryDataInfoData as String: gainData,
    kCGImageAuxiliaryDataInfoDataDescription as String: gainDesc as CFDictionary,
    kCGImageAuxiliaryDataInfoMetadata as String: gainMeta,
]

guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, UTType.heic.identifier as CFString, 1, nil) else {
    fail("无法创建 HEIC 目标")
}
CGImageDestinationAddImageFromSource(dest, sdrSrc, 0, props as CFDictionary)
CGImageDestinationAddAuxiliaryDataInfo(dest, kCGImageAuxiliaryDataTypeHDRGainMap, auxInfo as CFDictionary)
guard CGImageDestinationFinalize(dest) else { fail("写 HEIC 失败") }
print("✅ 已写出: \(outURL.path)")

// ---- 6. 验证：读回确认带 gain map ----
if let vSrc = CGImageSourceCreateWithURL(outURL as CFURL, nil),
   CGImageSourceCopyAuxiliaryDataInfoAtIndex(vSrc, 0, kCGImageAuxiliaryDataTypeHDRGainMap) != nil {
    print("✅ 验证通过：文件带 HDR gain map")
} else {
    print("⚠️ 验证：未读回 gain map")
}
