//
//  DriverMonitor.swift
//  DriveSensAI
//

import Combine
import Foundation
import Vision

/// Processes front-camera frames and publishes driver attention state.
@MainActor
final class DriverMonitor: ObservableObject {
    @Published private(set) var attentionState: DriverAttentionState = .noFace
    @Published private(set) var cameraError: CameraError?
    @Published private(set) var isRunning = false

    private let cameraManager = CameraManager()

    /// Serializes “Vision busy” so frames are dropped instead of stacking.
    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false

    /// Wall-clock start of continuous looking-away; cleared when facing forward or no face.
    private var lookingAwayStartedAt: Date?

    /// Approximate yaw (radians) beyond which the driver is considered looking away.
    private nonisolated static let yawLookAwayThreshold: Double = 0.40

    /// Looking away must persist this long before `attentionState` becomes `.lookingAway`.
    private let lookingAwayAlertDuration: TimeInterval = 2.0

    func start() {
        cameraError = nil

        cameraManager.onFrame = { [weak self] pixelBuffer in
            self?.handleFrame(pixelBuffer)
        }

        cameraManager.requestAccessAndStart { [weak self] result in
            Task { @MainActor in
                guard let self else { return }
                switch result {
                case .success:
                    self.isRunning = true
                    self.cameraError = nil
                case .failure(let error):
                    self.isRunning = false
                    self.cameraError = error
                    self.attentionState = .noFace
                }
            }
        }
    }

    func stop() {
        cameraManager.onFrame = nil
        cameraManager.stop()
        isRunning = false
        lookingAwayStartedAt = nil
        attentionState = .noFace
        processingLock.lock()
        isProcessingFrame = false
        processingLock.unlock()
    }

    // MARK: - Frame intake

    /// Called on the camera frame queue (not the main thread).
    private nonisolated func handleFrame(_ pixelBuffer: CVPixelBuffer) {
        processingLock.lock()
        if isProcessingFrame {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        processingLock.unlock()

        analyze(pixelBuffer: pixelBuffer, yawThreshold: Self.yawLookAwayThreshold)
    }

    private nonisolated func analyze(pixelBuffer: CVPixelBuffer, yawThreshold: Double) {
        defer {
            processingLock.lock()
            isProcessingFrame = false
            processingLock.unlock()
        }

        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .leftMirrored,
            options: [:]
        )

        do {
            try handler.perform([request])
            let observations = request.results ?? []
            let instant = Self.classifyInstantState(
                observations: observations,
                yawThreshold: yawThreshold
            )

            Task { @MainActor in
                self.applyTemporalLogic(instantState: instant)
            }
        } catch {
            Task { @MainActor in
                self.lookingAwayStartedAt = nil
                self.attentionState = .noFace
            }
        }
    }

    /// Immediate classification from a single frame (before the 2s looking-away gate).
    nonisolated private static func classifyInstantState(
        observations: [VNFaceObservation],
        yawThreshold: Double
    ) -> DriverAttentionState {
        guard observations.count == 1, let face = observations.first else {
            return .noFace
        }

        // Prefer yaw when Vision provides pose; missing pose → treat as usable but unknown → attentive.
        guard let yawValue = face.yaw?.doubleValue else {
            return .attentive
        }

        if abs(yawValue) >= yawThreshold {
            return .lookingAway
        }
        return .attentive
    }

    private func applyTemporalLogic(instantState: DriverAttentionState) {
        switch instantState {
        case .noFace:
            lookingAwayStartedAt = nil
            attentionState = .noFace

        case .attentive:
            lookingAwayStartedAt = nil
            attentionState = .attentive

        case .lookingAway:
            let now = Date()
            if lookingAwayStartedAt == nil {
                lookingAwayStartedAt = now
            }

            if let started = lookingAwayStartedAt,
               now.timeIntervalSince(started) >= lookingAwayAlertDuration {
                attentionState = .lookingAway
            } else {
                // Still within grace period — keep showing attentive.
                attentionState = .attentive
            }
        }
    }
}
