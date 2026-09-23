//
//  DrowsinessFeatureSample.swift
//  DriveSensAI
//

import Foundation

/// One Apple Vision–derived feature row matching the Python training CSV schema.
struct DrowsinessFeatureSample: Equatable, Sendable {
    /// Wall-clock milliseconds since epoch (UTC).
    let timestampMs: Int64
    /// Values in `DrowsinessFeatureContract.featureNames` order.
    let values: [Double]

    var isFinite: Bool {
        values.count == DrowsinessFeatureContract.featureCount
            && values.allSatisfy { $0.isFinite }
    }
}

/// Exact feature contract from `best_loss.pt` / `DROWSINESS_FEATURE_NAMES`.
enum DrowsinessFeatureContract {
    static let featureNames: [String] = [
        "face_detected",
        "vision_confidence",
        "yaw",
        "pitch",
        "roll",
        "left_eye_valid",
        "right_eye_valid",
        "left_eye_aspect_ratio",
        "right_eye_aspect_ratio",
        "left_eyelid_gap",
        "right_eyelid_gap",
        "left_pupil_x",
        "left_pupil_y",
        "right_pupil_x",
        "right_pupil_y",
        "mouth_valid",
        "mouth_aspect_ratio",
        "inner_lip_gap",
        "inner_mouth_area",
        "hand_detected",
        "hand_confidence",
        "hand_near_mouth",
    ]

    static var featureCount: Int { featureNames.count }

    /// Camera target FPS in `MultiCamManager` (not stored in the checkpoint).
    static let samplingRateHz: Double = 15.0

    /// Temporal window from `checkpoint["window_size"]`.
    static let windowFrames: Int = 5

    static let schemaVersion: Int = 1
}
