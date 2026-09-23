//
//  LaneDetectionService.swift
//  DriveSensAI
//
//  Experimental geometric lane tracker from MultiCam rear frames.
//  Not steering control — visual tracking + drift warning only.
//

import Combine
import CoreImage
import CoreVideo
import Foundation
import Vision

/// Consumes existing MultiCam rear `CVPixelBuffer`s (~5 Hz) and publishes lane assist state.
@MainActor
final class LaneDetectionService: ObservableObject {
    @Published private(set) var result: LaneTrackingResult = .empty
    @Published private(set) var isRunning = false

    #if DEBUG
    @Published private(set) var debugSnapshot = LaneDebugSnapshot()
    #endif

    // MARK: - Tunable constants

    /// Analysis rate target.
    nonisolated static let minInterval: TimeInterval = 0.20 // ~5 Hz

    /// Vision orientation — MultiCam rotates rear 90° for portrait, no mirror → upright buffers.
    nonisolated static let visionOrientation: CGImagePropertyOrientation = .up

    // Trapezoidal ROI in top-left normalized image coords (y=0 top, y=1 bottom).
    nonisolated static let roiTopY: CGFloat = 0.45
    nonisolated static let roiBottomY: CGFloat = 0.95
    nonisolated static let roiTopLeftX: CGFloat = 0.35
    nonisolated static let roiTopRightX: CGFloat = 0.65
    nonisolated static let roiBottomLeftX: CGFloat = 0.05
    nonisolated static let roiBottomRightX: CGFloat = 0.95

    /// Near-field row used to estimate lane width / center.
    nonisolated static let evaluationY: CGFloat = 0.85

    nonisolated static let vehicleCenterX: CGFloat = 0.50

    nonisolated static let emaAlpha: CGFloat = 0.30

    nonisolated static let confidenceAvailableThreshold: CGFloat = 0.45

    nonisolated static let driftEnterOffset: CGFloat = 0.55
    nonisolated static let driftExitOffset: CGFloat = 0.40
    nonisolated static let driftEnterDuration: TimeInterval = 0.70
    nonisolated static let driftExitDuration: TimeInterval = 0.40

    /// Lane-departure WARNING only at/above this GPS speed (MPH).
    nonisolated static let warningMinSpeedMPH: Double = 20.0

    nonisolated static let minLaneWidth: CGFloat = 0.18
    nonisolated static let maxLaneWidth: CGFloat = 0.85
    nonisolated static let minSegmentLength: CGFloat = 0.06
    nonisolated static let maxAbsSlopeForHorizontalReject: CGFloat = 0.12 // dy/dx ~ flat

    // MARK: - Runtime

    private let processingQueue = DispatchQueue(label: "com.drivesensai.lane.vision", qos: .userInitiated)
    private let processingLock = NSLock()
    private nonisolated(unsafe) var isProcessingFrame = false
    private nonisolated(unsafe) var lastAnalysisUptime: TimeInterval = 0
    private nonisolated(unsafe) var timingLogCounter = 0
    private nonisolated(unsafe) var didLogStartup = false

    /// Latest GPS speed for warning gate (updated from DriveView / SpeedMonitor).
    private var latestSpeedMPH: Double?
    private var latestSpeedReliable = false

    // Smoothed telemetry (MainActor)
    private var smoothLeftX: CGFloat?
    private var smoothRightX: CGFloat?
    private var smoothOffset: CGFloat = 0
    private var smoothConfidence: CGFloat = 0
    private var publishedState: LaneAssistState = .unavailable

    private var candidateDriftSide: LaneAssistState? // .driftingLeft / .driftingRight only
    private var candidateDriftSince: TimeInterval?
    private var centeredSince: TimeInterval?

    private var previousStateForLog: LaneAssistState = .unavailable

    func beginExternalFrameProcessing() {
        isRunning = true
        resetTracking()
        #if DEBUG
        if !didLogStartup {
            didLogStartup = true
            print("[Lane] rear frame processing active 5Hz")
        }
        #else
        if !didLogStartup {
            didLogStartup = true
            print("[Lane] rear frame processing active 5Hz")
        }
        #endif
    }

    func endExternalFrameProcessing() {
        isRunning = false
        resetTracking()
        result = .empty
    }

    /// Call when GPS speed updates (does not re-run Vision).
    func updateSpeed(mph: Double?, reliable: Bool) {
        latestSpeedMPH = mph
        latestSpeedReliable = reliable
        // Re-evaluate warning eligibility if currently drifting / tracking.
        if isRunning {
            republishWithCurrentSpeed()
        }
    }

    /// MultiCam rear callback — must return quickly; analysis is async on `processingQueue`.
    nonisolated func processExternalFrame(_ pixelBuffer: CVPixelBuffer) {
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastAnalysisUptime < Self.minInterval {
            return
        }

        processingLock.lock()
        if isProcessingFrame {
            processingLock.unlock()
            return
        }
        isProcessingFrame = true
        lastAnalysisUptime = now
        processingLock.unlock()

        // CIImage retains the buffer for async analysis (no manual CF retain).
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        processingQueue.async { [weak self] in
            defer {
                self?.processingLock.lock()
                self?.isProcessingFrame = false
                self?.processingLock.unlock()
            }
            guard let self else { return }
            let started = ProcessInfo.processInfo.systemUptime
            let raw = self.analyze(ciImage: ciImage)
            let elapsedMs = (ProcessInfo.processInfo.systemUptime - started) * 1000
            self.timingLogCounter += 1
            if self.timingLogCounter % 15 == 1 {
                print(String(format: "[Lane] analysis %.1f ms", elapsedMs))
            }
            Task { @MainActor in
                self.ingest(raw: raw, uptime: ProcessInfo.processInfo.systemUptime)
            }
        }
    }

    // MARK: - Analyze (background)

    private struct RawLaneSample {
        var leftX: CGFloat?
        var rightX: CGFloat?
        var confidence: CGFloat
    }

    nonisolated private func analyze(ciImage: CIImage) -> RawLaneSample {
        let mono = ciImage.applyingFilter("CIPhotoEffectMono", parameters: [:])
        let contrast = mono.applyingFilter("CIColorControls", parameters: [
            kCIInputContrastKey: 1.45,
            kCIInputBrightnessKey: 0.02,
            kCIInputSaturationKey: 0.0
        ])

        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.0
        request.detectsDarkOnLight = false // light lane paint on darker asphalt
        request.maximumImageDimension = 512

        let handler = VNImageRequestHandler(
            ciImage: contrast,
            orientation: Self.visionOrientation,
            options: [:]
        )

        do {
            try handler.perform([request])
        } catch {
            return RawLaneSample(leftX: nil, rightX: nil, confidence: 0)
        }

        guard let observation = request.results?.first as? VNContoursObservation else {
            return RawLaneSample(leftX: nil, rightX: nil, confidence: 0)
        }

        var leftCandidates: [(xAtEval: CGFloat, score: CGFloat, length: CGFloat)] = []
        var rightCandidates: [(xAtEval: CGFloat, score: CGFloat, length: CGFloat)] = []

        let contourCount = observation.contourCount
        for index in 0..<contourCount {
            guard let contour = try? observation.contour(at: index) else { continue }
            let points = contour.normalizedPoints
            guard points.count >= 4 else { continue }

            var xs: [CGFloat] = []
            var ys: [CGFloat] = []
            xs.reserveCapacity(points.count)
            ys.reserveCapacity(points.count)

            for i in 0..<points.count {
                // Vision normalized: origin bottom-left → convert to top-left to match ROI constants.
                let vx = CGFloat(points[i].x)
                let vyBottomOrigin = CGFloat(points[i].y)
                let yTopOrigin = 1.0 - vyBottomOrigin
                guard Self.pointInROI(x: vx, y: yTopOrigin) else { continue }
                xs.append(vx)
                ys.append(yTopOrigin)
            }

            guard xs.count >= 4 else { continue }

            guard let xFit = Self.linearFitXGivenY(xs: xs, ys: ys) else { continue }

            let y0 = ys.min()!
            let y1 = ys.max()!
            let length = abs(y1 - y0)
            guard length >= Self.minSegmentLength else { continue }

            let x0 = xFit.a * y0 + xFit.b
            let x1 = xFit.a * y1 + xFit.b
            let dx = x1 - x0
            let dy = y1 - y0
            // Reject nearly horizontal in image space.
            if abs(dy) < 1e-4 { continue }
            if abs(dx) > 1e-4 && abs(dy / dx) < Self.maxAbsSlopeForHorizontalReject {
                continue
            }

            let xAtEval = xFit.a * Self.evaluationY + xFit.b
            guard xAtEval > 0.02, xAtEval < 0.98 else { continue }

            let midX = (x0 + x1) * 0.5
            // Inward toward vanishing (smaller y): left a≤0, right a≥0 in x = a*y + b (y down).
            let inward: Bool
            if midX < 0.50 {
                inward = xFit.a < 0.05
            } else {
                inward = xFit.a > -0.05
            }

            let score = length * (inward ? 1.4 : 0.6) * CGFloat(min(xs.count, 40)) / 40.0

            if midX < 0.50, xAtEval < 0.55 {
                leftCandidates.append((xAtEval, score, length))
            } else if midX >= 0.50, xAtEval > 0.45 {
                rightCandidates.append((xAtEval, score, length))
            }
        }

        let bestLeft = leftCandidates.max(by: { $0.score < $1.score })
        let bestRight = rightCandidates.max(by: { $0.score < $1.score })

        var confidence: CGFloat = 0
        if let l = bestLeft, let r = bestRight, r.xAtEval > l.xAtEval {
            let width = r.xAtEval - l.xAtEval
            let widthOK = width >= Self.minLaneWidth && width <= Self.maxLaneWidth
            let lengthScore = min(1.0, (l.length + r.length) / 0.35)
            let widthScore: CGFloat = widthOK ? 1.0 : 0.25
            confidence = 0.35 * lengthScore + 0.45 * widthScore + 0.20 * min(1.0, (l.score + r.score) / 8.0)
            if !widthOK { confidence *= 0.5 }
            return RawLaneSample(leftX: l.xAtEval, rightX: r.xAtEval, confidence: min(1.0, confidence))
        }

        if bestLeft != nil || bestRight != nil {
            confidence = 0.22
        }
        return RawLaneSample(leftX: bestLeft?.xAtEval, rightX: bestRight?.xAtEval, confidence: confidence)
    }

    // MARK: - Geometry helpers

    nonisolated private static func pointInROI(x: CGFloat, y: CGFloat) -> Bool {
        guard y >= roiTopY, y <= roiBottomY else { return false }
        let t = (y - roiTopY) / max(roiBottomY - roiTopY, 1e-6)
        let left = roiTopLeftX + (roiBottomLeftX - roiTopLeftX) * t
        let right = roiTopRightX + (roiBottomRightX - roiTopRightX) * t
        return x >= left && x <= right
    }

    nonisolated private static func linearFitXGivenY(xs: [CGFloat], ys: [CGFloat]) -> (a: CGFloat, b: CGFloat)? {
        // x = a*y + b
        let n = CGFloat(ys.count)
        guard n >= 2 else { return nil }
        let sumY = ys.reduce(0, +)
        let sumX = xs.reduce(0, +)
        let sumYY = ys.reduce(0) { $0 + $1 * $1 }
        let sumYX = zip(ys, xs).reduce(CGFloat(0)) { $0 + $1.0 * $1.1 }
        let denom = n * sumYY - sumY * sumY
        guard abs(denom) > 1e-8 else { return nil }
        let a = (n * sumYX - sumY * sumX) / denom
        let b = (sumX - a * sumY) / n
        return (a, b)
    }

    // MARK: - Temporal / publish (MainActor)

    private func resetTracking() {
        smoothLeftX = nil
        smoothRightX = nil
        smoothOffset = 0
        smoothConfidence = 0
        publishedState = .unavailable
        candidateDriftSide = nil
        candidateDriftSince = nil
        centeredSince = nil
        previousStateForLog = .unavailable
        #if DEBUG
        debugSnapshot = LaneDebugSnapshot()
        #endif
    }

    private func ingest(raw: RawLaneSample, uptime: TimeInterval) {
        guard isRunning else { return }

        // EMA confidence
        smoothConfidence = Self.emaAlpha * raw.confidence + (1 - Self.emaAlpha) * smoothConfidence

        if let lx = raw.leftX {
            smoothLeftX = smoothLeftX.map { Self.emaAlpha * lx + (1 - Self.emaAlpha) * $0 } ?? lx
        }
        if let rx = raw.rightX {
            smoothRightX = smoothRightX.map { Self.emaAlpha * rx + (1 - Self.emaAlpha) * $0 } ?? rx
        }

        // Decay when a side is missing this frame.
        if raw.leftX == nil { smoothConfidence *= 0.92 }
        if raw.rightX == nil { smoothConfidence *= 0.92 }

        var offset: CGFloat = 0
        var laneCenter: CGFloat?
        if let l = smoothLeftX, let r = smoothRightX, r > l {
            let width = r - l
            let center = (l + r) * 0.5
            laneCenter = center
            let half = max(width * 0.5, 1e-3)
            // centerError > 0 when lane center is left of vehicle → vehicle is right of lane center.
            let centerError = Self.vehicleCenterX - center
            offset = centerError / half
            offset = max(-1.25, min(1.25, offset))
        } else {
            smoothConfidence *= 0.85
        }

        smoothOffset = Self.emaAlpha * offset + (1 - Self.emaAlpha) * smoothOffset
        smoothOffset = max(-1.2, min(1.2, smoothOffset))

        let nextState = resolveState(uptime: uptime)

        #if DEBUG
        debugSnapshot = LaneDebugSnapshot(
            leftX: smoothLeftX,
            rightX: smoothRightX,
            laneCenterX: laneCenter,
            confidence: smoothConfidence,
            lateralOffset: smoothOffset
        )
        #endif

        logTransitions(next: nextState)

        publishedState = nextState
        result = LaneTrackingResult(
            state: nextState,
            lateralOffset: smoothOffset,
            confidence: smoothConfidence,
            leftLaneBottomX: smoothLeftX,
            rightLaneBottomX: smoothRightX,
            laneCenterX: laneCenter
        )
    }

    private func resolveState(uptime: TimeInterval) -> LaneAssistState {
        if smoothConfidence < Self.confidenceAvailableThreshold
            || smoothLeftX == nil
            || smoothRightX == nil {
            candidateDriftSide = nil
            candidateDriftSince = nil
            centeredSince = nil
            return .unavailable
        }

        let absOff = abs(smoothOffset)
        let warningSpeedOK =
            latestSpeedReliable
            && (latestSpeedMPH ?? 0) >= Self.warningMinSpeedMPH

        // Hysteresis clear toward tracking.
        if publishedState == .driftingLeft || publishedState == .driftingRight {
            if absOff < Self.driftExitOffset {
                if centeredSince == nil { centeredSince = uptime }
                if let started = centeredSince,
                   uptime - started >= Self.driftExitDuration {
                    candidateDriftSide = nil
                    candidateDriftSince = nil
                    centeredSince = nil
                    #if DEBUG
                    print("[Lane] centered")
                    #endif
                    return .tracking
                }
                return publishedState
            } else {
                centeredSince = nil
                return publishedState
            }
        }

        // Enter drift only with speed gate + persistence.
        if warningSpeedOK, absOff >= Self.driftEnterOffset {
            let side: LaneAssistState = smoothOffset <= -Self.driftEnterOffset
                ? .driftingLeft
                : .driftingRight

            if candidateDriftSide != side {
                candidateDriftSide = side
                candidateDriftSince = uptime
                #if DEBUG
                print(side == .driftingLeft ? "[Lane] candidate driftLeft" : "[Lane] candidate driftRight")
                #endif
            } else if let started = candidateDriftSince,
                      uptime - started >= Self.driftEnterDuration {
                return side
            }
            return .tracking
        }

        candidateDriftSide = nil
        candidateDriftSince = nil
        centeredSince = nil
        return .tracking
    }

    private func republishWithCurrentSpeed() {
        let uptime = ProcessInfo.processInfo.systemUptime
        // If speed drops below gate while drifting, demote to tracking (keep visual offset).
        if (publishedState == .driftingLeft || publishedState == .driftingRight) {
            let warningSpeedOK =
                latestSpeedReliable
                && (latestSpeedMPH ?? 0) >= Self.warningMinSpeedMPH
            if !warningSpeedOK {
                publishedState = .tracking
                candidateDriftSide = nil
                candidateDriftSince = nil
                centeredSince = nil
                result = LaneTrackingResult(
                    state: .tracking,
                    lateralOffset: result.lateralOffset,
                    confidence: result.confidence,
                    leftLaneBottomX: result.leftLaneBottomX,
                    rightLaneBottomX: result.rightLaneBottomX,
                    laneCenterX: result.laneCenterX
                )
                #if DEBUG
                print("[Lane] drift cleared (speed gate)")
                #endif
                return
            }
        }
        let next = resolveState(uptime: uptime)
        if next != publishedState {
            logTransitions(next: next)
            publishedState = next
            result = LaneTrackingResult(
                state: next,
                lateralOffset: result.lateralOffset,
                confidence: result.confidence,
                leftLaneBottomX: result.leftLaneBottomX,
                rightLaneBottomX: result.rightLaneBottomX,
                laneCenterX: result.laneCenterX
            )
        }
    }

    private func logTransitions(next: LaneAssistState) {
        if previousStateForLog == .unavailable, next == .tracking {
            print(String(format: "[Lane] acquired confidence=%.2f", Double(smoothConfidence)))
        }
        if previousStateForLog != .unavailable, next == .unavailable {
            print(String(format: "[Lane] lost confidence=%.2f", Double(smoothConfidence)))
        }
        if next == .driftingLeft, previousStateForLog != .driftingLeft {
            print(String(format: "[Lane] DRIFT_LEFT offset=%.2f", Double(smoothOffset)))
        }
        if next == .driftingRight, previousStateForLog != .driftingRight {
            print(String(format: "[Lane] DRIFT_RIGHT offset=%.2f", Double(smoothOffset)))
        }
        #if DEBUG
        // Occasional offset sample while tracking.
        if next == .tracking, timingSampleShouldLog() {
            print(String(format: "[Lane] offset=%.2f state=tracking", Double(smoothOffset)))
        }
        #endif
        previousStateForLog = next
    }

    private var debugOffsetLogCounter = 0
    private func timingSampleShouldLog() -> Bool {
        debugOffsetLogCounter += 1
        return debugOffsetLogCounter % 10 == 0
    }
}
