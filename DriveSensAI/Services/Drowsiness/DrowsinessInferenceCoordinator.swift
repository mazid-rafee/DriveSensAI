//
//  DrowsinessInferenceCoordinator.swift
//  DriveSensAI
//

import Combine
import Foundation

/// Buffers ~15 FPS feature rows and POSTs a T=5 window every 400 ms.
/// Publishes a wake-up flag after two consecutive `"close"` labels.
final class DrowsinessInferenceCoordinator: ObservableObject {
    /// Latest remote prediction (logging + UI).
    @Published private(set) var latestPrediction: DrowsinessPredictResponse?
    /// True after two consecutive successful replies with `label == "close"`.
    @Published private(set) var isWakeUpAlertActive = false

    private let client = DrowsinessAPIClient()
    private let lock = NSLock()

    private var sessionID = UUID().uuidString
    private var sequenceID = 0
    private var buffer: [DrowsinessFeatureSample] = []
    private var isRunning = false
    private var requestInFlight = false
    private var sendTimer: Timer?
    private var activeTask: Task<Void, Never>?
    private var consecutiveCloseCount = 0

    private let windowFrames = DrowsinessFeatureContract.windowFrames
    private let maxBuffer = DrowsinessFeatureContract.windowFrames * 4
    private let sendInterval: TimeInterval = 0.4
    private let wakeUpCloseThreshold = 5
    private static let closeLabel = "close"
    private static let undefinedLabel = "undefined"

    func start() {
        lock.lock()
        sessionID = UUID().uuidString
        sequenceID = 0
        buffer.removeAll(keepingCapacity: true)
        requestInFlight = false
        consecutiveCloseCount = 0
        isRunning = true
        lock.unlock()

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.latestPrediction = nil
            self.isWakeUpAlertActive = false
            self.sendTimer?.invalidate()
            let timer = Timer.scheduledTimer(withTimeInterval: self.sendInterval, repeats: true) { [weak self] _ in
                self?.tickSend()
            }
            RunLoop.main.add(timer, forMode: .common)
            self.sendTimer = timer
        }
    }

    func stop() {
        lock.lock()
        isRunning = false
        buffer.removeAll(keepingCapacity: false)
        requestInFlight = false
        consecutiveCloseCount = 0
        lock.unlock()

        activeTask?.cancel()
        activeTask = nil
        DispatchQueue.main.async { [weak self] in
            self?.latestPrediction = nil
            self?.isWakeUpAlertActive = false
            self?.sendTimer?.invalidate()
            self?.sendTimer = nil
        }
    }

    /// Called from the camera / Vision path (any queue).
    nonisolated func ingest(_ sample: DrowsinessFeatureSample) {
        guard sample.isFinite else { return }
        lock.lock()
        defer { lock.unlock() }
        guard isRunning else { return }
        buffer.append(sample)
        if buffer.count > maxBuffer {
            buffer.removeFirst(buffer.count - maxBuffer)
        }
    }

    private func tickSend() {
        lock.lock()
        guard isRunning, !requestInFlight, buffer.count >= windowFrames else {
            lock.unlock()
            return
        }
        let window = Array(buffer.suffix(windowFrames))
        sequenceID += 1
        let seq = sequenceID
        let session = sessionID
        requestInFlight = true
        lock.unlock()

        let request = DrowsinessPredictRequest(
            schemaVersion: DrowsinessFeatureContract.schemaVersion,
            sessionID: session,
            sequenceID: seq,
            sentAtUTC: Date(),
            samplingRateHz: DrowsinessFeatureContract.samplingRateHz,
            featureNames: DrowsinessFeatureContract.featureNames,
            samples: window.map {
                DrowsinessAPISample(timestampMs: $0.timestampMs, values: $0.values)
            }
        )

        let started = Date()
        activeTask = Task { [weak self] in
            guard let self else { return }
            do {
                let response = try await client.predict(request)
                let roundTripMs = Date().timeIntervalSince(started) * 1000.0
                await self.handleSuccess(
                    expectedSession: session,
                    expectedSequence: seq,
                    response: response,
                    roundTripMs: roundTripMs
                )
            } catch is CancellationError {
                await self.clearInFlight()
            } catch {
                await self.handleFailure(
                    expectedSession: session,
                    expectedSequence: seq,
                    error: error
                )
            }
        }
    }

    @MainActor
    private func handleSuccess(
        expectedSession: String,
        expectedSequence: Int,
        response: DrowsinessPredictResponse,
        roundTripMs: Double
    ) {
        lock.lock()
        let sessionMatches = sessionID == expectedSession
        let sequenceMatches = sequenceID == expectedSequence
        let stillRunning = isRunning
        requestInFlight = false
        lock.unlock()

        guard stillRunning, sessionMatches else { return }
        // Ignore stale responses if a newer sequence was already issued.
        guard response.sessionID == expectedSession,
              response.sequenceID == expectedSequence,
              sequenceMatches else {
            return
        }

        latestPrediction = response
        updateWakeUpStreak(label: response.label)
        print(
            "[DrowsinessRemote] session=\(response.sessionID) seq=\(response.sequenceID) "
                + "label=\(response.label) confidence=\(String(format: "%.4f", response.confidence)) "
                + "model=\(response.modelVersion) server_ms=\(String(format: "%.1f", response.inferenceLatencyMs)) "
                + "round_trip_ms=\(String(format: "%.1f", roundTripMs))"
        )
    }

    @MainActor
    private func updateWakeUpStreak(label: String) {
        lock.lock()
        if label == Self.undefinedLabel {
            consecutiveCloseCount += 1
        } else {
            consecutiveCloseCount = 0
        }
        let active = consecutiveCloseCount >= wakeUpCloseThreshold
        lock.unlock()
        isWakeUpAlertActive = active
    }

    @MainActor
    private func handleFailure(
        expectedSession: String,
        expectedSequence: Int,
        error: Error
    ) {
        lock.lock()
        requestInFlight = false
        // Failures are not labeled replies — clear streak so wake-up never trips on errors.
        consecutiveCloseCount = 0
        lock.unlock()
        isWakeUpAlertActive = false
        // Network / server failure must never become a drowsiness-positive UI signal.
        print(
            "[DrowsinessRemote][ERROR] session=\(expectedSession) seq=\(expectedSequence) "
                + "error=\(error.localizedDescription)"
        )
    }

    @MainActor
    private func clearInFlight() {
        lock.lock()
        requestInFlight = false
        lock.unlock()
    }
}
