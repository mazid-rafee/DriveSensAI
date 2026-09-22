//
//  DrowsinessFeatureExtractor.swift
//  DriveSensAI
//

import Foundation
import Vision

/// Builds training-compatible feature rows from Apple Vision face landmarks + hand pose.
///
/// Training CSVs (`*_rgb_face.apple_drowsiness.csv`) are consumed as raw floats by
/// `pre_process/dataloader.py` — that package does **not** recompute EAR/MAR.
/// Geometry below follows standard Vision normalized image coordinates (origin bottom-left)
/// so column *names* and missing-value behavior (0.0 + validity flags) match training.
enum DrowsinessFeatureExtractor {
    /// Image-normalized distance below which a hand joint is "near mouth".
    private static let handNearMouthThreshold: Double = 0.18

    static func makeSample(
        faces: [VNFaceObservation],
        hands: [VNHumanHandPoseObservation],
        timestamp: Date = Date()
    ) -> DrowsinessFeatureSample {
        let timestampMs = Int64((timestamp.timeIntervalSince1970 * 1000.0).rounded())
        var values = Array(repeating: 0.0, count: DrowsinessFeatureContract.featureCount)

        let face = faces.max(by: {
            ($0.boundingBox.width * $0.boundingBox.height)
                < ($1.boundingBox.width * $1.boundingBox.height)
        })

        if let face {
            values[0] = 1.0
            values[1] = Double(face.confidence)
            values[2] = face.yaw?.doubleValue ?? 0.0
            values[3] = face.pitch?.doubleValue ?? 0.0
            values[4] = face.roll?.doubleValue ?? 0.0

            let landmarks = face.landmarks
            let box = face.boundingBox

            if let leftEye = landmarks?.leftEye, leftEye.pointCount >= 2 {
                values[5] = 1.0
                let metrics = eyeMetrics(region: leftEye, faceBox: box)
                values[7] = metrics.aspectRatio
                values[9] = metrics.eyelidGap
            }
            if let rightEye = landmarks?.rightEye, rightEye.pointCount >= 2 {
                values[6] = 1.0
                let metrics = eyeMetrics(region: rightEye, faceBox: box)
                values[8] = metrics.aspectRatio
                values[10] = metrics.eyelidGap
            }

            if let pupil = landmarks?.leftPupil, pupil.pointCount >= 1 {
                let p = imagePoint(pupil.normalizedPoints[0], faceBox: box)
                values[11] = p.x
                values[12] = p.y
            } else if let leftEye = landmarks?.leftEye, leftEye.pointCount >= 1 {
                let p = centroid(region: leftEye, faceBox: box)
                values[11] = p.x
                values[12] = p.y
            }

            if let pupil = landmarks?.rightPupil, pupil.pointCount >= 1 {
                let p = imagePoint(pupil.normalizedPoints[0], faceBox: box)
                values[13] = p.x
                values[14] = p.y
            } else if let rightEye = landmarks?.rightEye, rightEye.pointCount >= 1 {
                let p = centroid(region: rightEye, faceBox: box)
                values[13] = p.x
                values[14] = p.y
            }

            let outer = landmarks?.outerLips
            let inner = landmarks?.innerLips
            if let outer, outer.pointCount >= 2 {
                values[15] = 1.0
                let mouth = mouthMetrics(outer: outer, inner: inner, faceBox: box)
                values[16] = mouth.aspectRatio
                values[17] = mouth.innerLipGap
                values[18] = mouth.innerMouthArea
            }
        }

        if let hand = hands.max(by: { $0.confidence < $1.confidence }) {
            values[19] = 1.0
            values[20] = Double(hand.confidence)
            if let mouthCenter = mouthCenterImagePoint(from: face),
               let handPoint = primaryHandPoint(hand) {
                let dx = handPoint.x - mouthCenter.x
                let dy = handPoint.y - mouthCenter.y
                let distance = (dx * dx + dy * dy).squareRoot()
                values[21] = distance <= handNearMouthThreshold ? 1.0 : 0.0
            }
        }

        // Training CSV path maps empty/NaN → 0.0; never emit non-finite values.
        for index in values.indices where !values[index].isFinite {
            values[index] = 0.0
        }

        return DrowsinessFeatureSample(timestampMs: timestampMs, values: values)
    }

    // MARK: - Geometry

    private struct EyeMetrics {
        var aspectRatio: Double
        var eyelidGap: Double
    }

    private struct MouthMetrics {
        var aspectRatio: Double
        var innerLipGap: Double
        var innerMouthArea: Double
    }

    private static func eyeMetrics(
        region: VNFaceLandmarkRegion2D,
        faceBox: CGRect
    ) -> EyeMetrics {
        let points = region.normalizedPoints.map { imagePoint($0, faceBox: faceBox) }
        guard let xsMin = points.map(\.x).min(),
              let xsMax = points.map(\.x).max(),
              let ysMin = points.map(\.y).min(),
              let ysMax = points.map(\.y).max() else {
            return EyeMetrics(aspectRatio: 0, eyelidGap: 0)
        }
        let width = max(xsMax - xsMin, 1e-6)
        let height = max(ysMax - ysMin, 0)
        return EyeMetrics(aspectRatio: height / width, eyelidGap: height)
    }

    private static func mouthMetrics(
        outer: VNFaceLandmarkRegion2D,
        inner: VNFaceLandmarkRegion2D?,
        faceBox: CGRect
    ) -> MouthMetrics {
        let outerPoints = outer.normalizedPoints.map { imagePoint($0, faceBox: faceBox) }
        let xs = outerPoints.map(\.x)
        let ys = outerPoints.map(\.y)
        let width = max((xs.max() ?? 0) - (xs.min() ?? 0), 1e-6)
        let height = max((ys.max() ?? 0) - (ys.min() ?? 0), 0)
        let aspect = height / width

        var innerGap = 0.0
        var innerArea = 0.0
        if let inner, inner.pointCount >= 2 {
            let innerPoints = inner.normalizedPoints.map { imagePoint($0, faceBox: faceBox) }
            let iys = innerPoints.map(\.y)
            innerGap = max((iys.max() ?? 0) - (iys.min() ?? 0), 0)
            innerArea = polygonArea(innerPoints)
        }
        return MouthMetrics(
            aspectRatio: aspect,
            innerLipGap: innerGap,
            innerMouthArea: innerArea
        )
    }

    private static func centroid(
        region: VNFaceLandmarkRegion2D,
        faceBox: CGRect
    ) -> CGPoint {
        let points = region.normalizedPoints.map { imagePoint($0, faceBox: faceBox) }
        guard !points.isEmpty else { return .zero }
        let sum = points.reduce(CGPoint.zero) { CGPoint(x: $0.x + $1.x, y: $0.y + $1.y) }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }

    /// Landmark points are relative to the face bounding box; convert to image coords.
    private static func imagePoint(_ normalizedInFace: CGPoint, faceBox: CGRect) -> CGPoint {
        CGPoint(
            x: faceBox.origin.x + normalizedInFace.x * faceBox.width,
            y: faceBox.origin.y + normalizedInFace.y * faceBox.height
        )
    }

    private static func mouthCenterImagePoint(from face: VNFaceObservation?) -> CGPoint? {
        guard let face else { return nil }
        if let outer = face.landmarks?.outerLips, outer.pointCount >= 1 {
            return centroid(region: outer, faceBox: face.boundingBox)
        }
        let box = face.boundingBox
        return CGPoint(x: box.midX, y: box.minY + box.height * 0.25)
    }

    private static func primaryHandPoint(_ hand: VNHumanHandPoseObservation) -> CGPoint? {
        let candidates: [VNHumanHandPoseObservation.JointName] = [
            .wrist, .indexTip, .middleTip, .thumbTip, .ringTip, .littleTip,
        ]
        for name in candidates {
            if let point = try? hand.recognizedPoint(name), point.confidence > 0.2 {
                return CGPoint(x: point.location.x, y: point.location.y)
            }
        }
        return nil
    }

    private static func polygonArea(_ points: [CGPoint]) -> Double {
        guard points.count >= 3 else { return 0 }
        var sum: Double = 0
        for index in points.indices {
            let next = points[(index + 1) % points.count]
            sum += Double(points[index].x * next.y - next.x * points[index].y)
        }
        return abs(sum) * 0.5
    }
}
