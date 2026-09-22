//
//  SpeedMonitor.swift
//  DriveSensAI
//

import Combine
import CoreLocation
import Foundation

/// Publishes smoothed GPS vehicle speed in MPH for the driving dashboard.
/// Uses its own CLLocationManager; does not share Maps / Places / Navigation managers.
@MainActor
final class SpeedMonitor: NSObject, ObservableObject {
    @Published private(set) var speedMPH: Double?
    @Published private(set) var hasReliableSpeed = false

    // MARK: - Experimental tuning

    /// Reject updates whose horizontal speed accuracy is worse than this (meters/second).
    nonisolated static let maximumSpeedAccuracyMetersPerSecond: CLLocationSpeedAccuracy = 3.0

    /// EMA blend for readable but responsive MPH.
    nonisolated static let smoothingAlpha: Double = 0.25

    /// Below this smoothed MPH, display 0 to avoid GPS creep while stopped.
    nonisolated static let stationaryDisplayThresholdMPH: Double = 1.0

    /// Drop smoothing history if no valid fix arrives for this long.
    nonisolated static let smoothingResetGap: TimeInterval = 5.0

    /// Ignore location fixes older than this.
    nonisolated static let maximumLocationAge: TimeInterval = 5.0

    nonisolated static let metersPerSecondToMPH: Double = 2.2369362920544

    private let manager = CLLocationManager()
    private var isMonitoring = false
    private var smoothedMPH: Double?
    private var lastValidMeasurementAt: Date?
    #if DEBUG
    private var lastAcceptedLogAt: Date?
    #endif

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        manager.activityType = .automotiveNavigation
        manager.distanceFilter = kCLDistanceFilterNone
        manager.pausesLocationUpdatesAutomatically = false
    }

    func start() {
        guard !isMonitoring else { return }
        isMonitoring = true

        switch manager.authorizationStatus {
        case .authorizedAlways, .authorizedWhenInUse:
            beginUpdating()
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .denied, .restricted:
            clearPublishedSpeed()
        @unknown default:
            clearPublishedSpeed()
        }
    }

    func stop() {
        guard isMonitoring else { return }
        isMonitoring = false
        manager.stopUpdatingLocation()
        smoothedMPH = nil
        lastValidMeasurementAt = nil
        clearPublishedSpeed()
    }

    private func beginUpdating() {
        manager.startUpdatingLocation()
    }

    private func clearPublishedSpeed() {
        speedMPH = nil
        hasReliableSpeed = false
    }

    // MARK: - Shared processing pipeline

    /// Validates and smooths a location fix. Used by live GPS and DEBUG injection.
    private func processLocation(_ location: CLLocation) {
        switch validate(location) {
        case .rejected(let reason):
            #if DEBUG
            print("[Speed] rejected \(reason)")
            #endif
            return
        case .accepted(let mph):
            applySmoothing(
                rawMPH: mph,
                speedMetersPerSecond: location.speed,
                speedAccuracy: location.speedAccuracy,
                at: location.timestamp
            )
        }
    }

    private func handleLocations(_ locations: [CLLocation]) {
        guard isMonitoring, let location = locations.last else { return }
        processLocation(location)
    }

    private enum ValidationResult {
        case accepted(mph: Double)
        case rejected(String)
    }

    private func validate(_ location: CLLocation) -> ValidationResult {
        let age = -location.timestamp.timeIntervalSinceNow
        if age > Self.maximumLocationAge || age < -1 {
            return .rejected("stale age=\(String(format: "%.1f", age))s")
        }

        if location.speed < 0 {
            return .rejected("speed<0")
        }

        if location.speedAccuracy < 0 {
            return .rejected("speedAccuracy<0")
        }

        if location.speedAccuracy > Self.maximumSpeedAccuracyMetersPerSecond {
            return .rejected(
                String(format: "speedAccuracy=%.1f", location.speedAccuracy)
            )
        }

        let mph = location.speed * Self.metersPerSecondToMPH
        return .accepted(mph: mph)
    }

    private func applySmoothing(
        rawMPH: Double,
        speedMetersPerSecond: CLLocationSpeed,
        speedAccuracy: CLLocationSpeedAccuracy,
        at timestamp: Date
    ) {
        if let last = lastValidMeasurementAt,
           timestamp.timeIntervalSince(last) > Self.smoothingResetGap
            || last.timeIntervalSince(timestamp) > Self.smoothingResetGap {
            smoothedMPH = nil
        }
        lastValidMeasurementAt = timestamp

        let alpha = Self.smoothingAlpha
        let next: Double
        if let previous = smoothedMPH {
            next = alpha * rawMPH + (1.0 - alpha) * previous
        } else {
            next = rawMPH
        }
        smoothedMPH = next

        let display: Double
        if next < Self.stationaryDisplayThresholdMPH {
            display = 0
        } else {
            display = next
        }

        speedMPH = display
        hasReliableSpeed = true

        #if DEBUG
        logAcceptedIfNeeded(
            metersPerSecond: speedMetersPerSecond,
            accuracy: speedAccuracy,
            rawMPH: rawMPH,
            smoothed: next,
            at: timestamp
        )
        #endif
    }

    #if DEBUG
    private func logAcceptedIfNeeded(
        metersPerSecond: CLLocationSpeed,
        accuracy: CLLocationSpeedAccuracy,
        rawMPH: Double,
        smoothed: Double,
        at timestamp: Date
    ) {
        if let last = lastAcceptedLogAt, timestamp.timeIntervalSince(last) < 1.0 {
            return
        }
        lastAcceptedLogAt = timestamp
        print(
            String(
                format: "[Speed] raw=%.2fm/s accuracy=%.1fm/s mph=%.1f smoothed=%.1f",
                metersPerSecond,
                accuracy,
                rawMPH,
                smoothed
            )
        )
    }
    #endif
}

extension SpeedMonitor: CLLocationManagerDelegate {
    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        Task { @MainActor in
            guard self.isMonitoring else { return }
            switch manager.authorizationStatus {
            case .authorizedAlways, .authorizedWhenInUse:
                self.beginUpdating()
            case .denied, .restricted:
                self.manager.stopUpdatingLocation()
                self.smoothedMPH = nil
                self.lastValidMeasurementAt = nil
                self.clearPublishedSpeed()
            case .notDetermined:
                break
            @unknown default:
                break
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            self.handleLocations(locations)
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        #if DEBUG
        print("[Speed] location error: \(error.localizedDescription)")
        #endif
    }
}
