//
//  DriveView.swift
//  DriveSensAI
//

import Combine
import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @StateObject private var roadDetector = RoadDetectionService()
    @StateObject private var roadRiskAnalyzer = RoadRiskAnalyzer()
    @StateObject private var pedestrianRiskAnalyzer = PedestrianRiskAnalyzer()
    @StateObject private var speedMonitor = SpeedMonitor()
    @StateObject private var laneDetector = LaneDetectionService()
    @StateObject private var alertManager = ADASAlertManager()
    @StateObject private var drowsinessRemote = DrowsinessInferenceCoordinator()

    /// Stable owner for the non-Observable MultiCamManager + published UI status.
    @StateObject private var multiCamOwner = MultiCamSessionOwner()

    /// Bridged from Google Navigation overspeed callbacks (unavailable outside guidance).
    @State private var speedingState: SpeedingState = .unavailable
    /// Raw Navigation SDK speeding fraction; `nil` outside guidance / after reset.
    @State private var percentageAboveLimit: CGFloat? = nil

    #if DEBUG
    @State private var showLaneDebugOverlay = false
    @State private var lastOverLimitLogKey: String = ""
    @State private var lastRoadDisplayLog: String = ""
    #endif

    var body: some View {
        ZStack(alignment: .top) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 8) {
                topBar

                // Map fills remaining height; telemetry stays compact so the map stays dominant.
                NavigationView(
                    speedingState: $speedingState,
                    percentageAboveLimit: $percentageAboveLimit
                )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                lowerTelemetry
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 16)
            .padding(.top, 6)
            .padding(.bottom, 8)

            // TEMP: pin wake banner above the map when closed is detected (5s hold).
            if drowsinessRemote.isWakeUpAlertActive {
                wakeUpDebugBanner
                    .padding(.horizontal, 16)
                    .padding(.top, 44)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .ignoresSafeArea(.keyboard, edges: .bottom)
        .preferredColorScheme(.dark)
        .animation(.easeInOut(duration: 0.2), value: driverMonitor.attentionState)
        .animation(.easeInOut(duration: 0.2), value: unifiedRoadDisplay)
        .animation(.easeInOut(duration: 0.2), value: laneDetector.result.state)
        .animation(.easeInOut(duration: 0.2), value: percentageAboveLimit)
        .animation(.easeInOut(duration: 0.2), value: speedMonitor.speedMPH)
        .animation(.easeInOut(duration: 0.2), value: warningBanner?.title)
        .animation(.easeInOut(duration: 0.2), value: drowsinessRemote.isWakeUpAlertActive)
        .animation(.easeInOut(duration: 0.2), value: drowsinessRemote.latestPrediction?.label)
        .animation(.easeInOut(duration: 0.2), value: wakeUpDebugBannerTitle)
        .onAppear {
            startMultiCamIfNeeded()
            speedMonitor.start()
            alertManager.start()
            drowsinessRemote.start()
            syncLaneSpeedGate()
            syncADASAlerts()
        }
        .onDisappear {
            alertManager.stop()
            drowsinessRemote.stop()
            stopMultiCam()
            speedMonitor.stop()
        }
        .onReceive(roadDetector.$detections) { detections in
            let timestamp = ProcessInfo.processInfo.systemUptime
            roadRiskAnalyzer.update(detections: detections, timestamp: timestamp)
            pedestrianRiskAnalyzer.update(detections: detections, timestamp: timestamp)
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: roadDetector.isModelReady) { _, ready in
            if !ready {
                roadRiskAnalyzer.reset()
                pedestrianRiskAnalyzer.reset()
            }
            syncADASAlerts()
        }
        .onChange(of: driverMonitor.attentionState) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: roadRiskAnalyzer.state) { _, _ in
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: pedestrianRiskAnalyzer.state) { _, _ in
            logUnifiedRoadDisplayIfNeeded()
            syncADASAlerts()
        }
        .onChange(of: laneDetector.result.state) { _, _ in
            syncADASAlerts()
        }
        .onChange(of: drowsinessRemote.isWakeUpAlertActive) { wasActive, isActive in
            syncADASAlerts()
            // Beep only when the banner becomes visible (rising edge).
            if isActive, !wasActive {
                alertManager.playWakeUpBeep()
            }
        }
        .onChange(of: speedMonitor.speedMPH) { _, _ in
            syncLaneSpeedGate()
            logOverLimitIfNeeded()
        }
        .onChange(of: speedMonitor.hasReliableSpeed) { _, _ in
            syncLaneSpeedGate()
        }
        .onChange(of: percentageAboveLimit) { _, _ in
            logOverLimitIfNeeded()
        }
    }

    /// Feeds centralized alert manager from explicit state changes only.
    private func syncADASAlerts() {
        alertManager.update(
            driverAttention: driverMonitor.attentionState,
            roadRisk: roadRiskAnalyzer.state,
            pedestrianRisk: pedestrianRiskAnalyzer.state,
            laneAssist: laneDetector.result.state,
            wakeUpAlert: drowsinessRemote.isWakeUpAlertActive
        )
    }

    private func syncLaneSpeedGate() {
        laneDetector.updateSpeed(
            mph: speedMonitor.speedMPH,
            reliable: speedMonitor.hasReliableSpeed
        )
    }

    /// Compact strip under the map: optional warning → speed | lane | over-limit → ADAS chips → footer.
    private var lowerTelemetry: some View {
        VStack(spacing: 6) {
            if let banner = warningBanner {
                WarningBannerView(title: banner.title, style: banner.style)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            instrumentClusterRow

            adasStatusRow

            onDeviceFooter
        }
    }

    /// Wake banner when closed eyes are detected (held 5s by the coordinator).
    private var wakeUpDebugBanner: some View {
        WarningBannerView(title: wakeUpDebugBannerTitle, style: .urgent)
            .accessibilityLabel(wakeUpDebugBannerTitle)
    }

    private var wakeUpDebugBannerTitle: String {
        if let prediction = drowsinessRemote.latestPrediction {
            let confidence = String(format: "%.0f%%", prediction.confidence * 100.0)
            return "Wake up · \(prediction.label) (\(confidence))"
        }
        return "Wake up · closed"
    }

    // MARK: - Top bar

    private var topBar: some View {
        HStack(alignment: .center) {
            Text("DriveSensAI")
                .font(.headline.weight(.semibold))
                .foregroundStyle(.primary)

            Spacer()

            liveIndicator
        }
    }

    private var liveIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(multiCamOwner.isActive ? Color.green : Color.orange)
                .frame(width: 8, height: 8)

            Text(multiCamOwner.isActive ? "LIVE" : (multiCamOwner.errorMessage == nil ? "STARTING" : "CAMERA ERROR"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(multiCamOwner.isActive ? Color.green : Color.orange)
        }
        .accessibilityLabel(multiCamOwner.isActive ? "Live" : "Camera error")
    }

    // MARK: - Instrument row: speed | lane | over-limit (compact, centered)

    private let laneRoadHeight: CGFloat = 46

    private var instrumentClusterRow: some View {
        HStack(alignment: .center, spacing: 16) {
            speedColumn
                .frame(maxWidth: .infinity, alignment: .trailing)
            laneAssistColumn
                .frame(width: 64)
            overLimitColumn
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(speedAccessibilityLabel)
    }

    private var speedColumn: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text("SPEED MPH")
                .font(.system(size: 10, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(.secondary)
            Text(speedDisplayText)
                .font(.system(size: 28, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(speedForeground)
        }
    }

    /// Right column — same horizontal layout as former LIMIT; shows MPH over posted limit.
    private var overLimitColumn: some View {
        HStack(alignment: .firstTextBaseline, spacing: 4) {
            Text(overLimitDisplayText)
                .font(.system(size: 28, weight: .semibold, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(.primary.opacity(0.92))
            Text("OVER LIMIT")
                .font(.system(size: 10, weight: .bold))
                .tracking(0.4)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var laneAssistColumn: some View {
        #if DEBUG
        LaneAssistView(
            result: laneDetector.result,
            roadHeight: laneRoadHeight,
            debugSnapshot: laneDetector.debugSnapshot,
            showDebugOverlay: showLaneDebugOverlay
        )
        .onLongPressGesture(minimumDuration: 0.6) {
            showLaneDebugOverlay.toggle()
        }
        #else
        LaneAssistView(result: laneDetector.result, roadHeight: laneRoadHeight)
        #endif
    }

    private var speedDisplayText: String {
        guard speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH else {
            return "--"
        }
        return "\(Int(mph.rounded()))"
    }

    /// Whole MPH over the posted limit from Nav SDK percentage + GPS speed.
    /// `p > 0` → `S * p / (1 + p)`; `p == 0` → `0`; `p < 0` / missing → `--`.
    private var overLimitDisplayText: String {
        guard let p = percentageAboveLimit else { return "--" }
        if p < 0 { return "--" }
        if p == 0 { return "0" }
        guard speedMonitor.hasReliableSpeed, let speed = speedMonitor.speedMPH else {
            return "--"
        }
        let over = speed * Double(p) / (1.0 + Double(p))
        return "\(Int(over.rounded()))"
    }

    /// Current speed only: deeper red as MPH over the limit increases.
    private var speedForeground: Color {
        guard speedMonitor.hasReliableSpeed,
              let mph = speedMonitor.speedMPH,
              let p = percentageAboveLimit,
              p > 0 else {
            return .primary
        }
        let over = mph * Double(p) / (1.0 + Double(p))
        guard over > 0 else { return .primary }

        // 0 mph over → soft red; ≥20 mph over → deep crimson.
        let t = min(1.0, over / 20.0)
        return Color(
            red: 0.92 - 0.22 * t,
            green: 0.28 - 0.24 * t,
            blue: 0.22 - 0.16 * t
        )
    }

    private var speedAccessibilityLabel: String {
        let speedPart: String
        if speedMonitor.hasReliableSpeed, let mph = speedMonitor.speedMPH {
            speedPart = "\(Int(mph.rounded())) miles per hour"
        } else {
            speedPart = "Speed unavailable"
        }
        switch overLimitDisplayText {
        case "--":
            return "\(speedPart), over limit unavailable"
        case "0":
            return "\(speedPart), not over limit"
        default:
            return "\(speedPart), \(overLimitDisplayText) over limit"
        }
    }

    private func logOverLimitIfNeeded() {
        #if DEBUG
        guard let p = percentageAboveLimit else {
            let key = "nil"
            guard key != lastOverLimitLogKey else { return }
            lastOverLimitLogKey = key
            print("[OverLimit] percentage unavailable")
            return
        }
        if p < 0 {
            let key = "neg"
            guard key != lastOverLimitLogKey else { return }
            lastOverLimitLogKey = key
            print("[OverLimit] percentage=-1 unavailable")
            return
        }
        let speed = speedMonitor.hasReliableSpeed ? speedMonitor.speedMPH : nil
        let overText: String
        if p == 0 {
            overText = "0"
        } else if let speed {
            overText = "\(Int((speed * Double(p) / (1.0 + Double(p))).rounded()))"
        } else {
            overText = "n/a"
        }
        let speedText = speed.map { String(format: "%.1f", $0) } ?? "n/a"
        let key = "\(speedText)|\(String(format: "%.4f", Double(p)))|\(overText)"
        guard key != lastOverLimitLogKey else { return }
        lastOverLimitLogKey = key
        if p == 0 {
            print("[OverLimit] speedMPH=\(speedText) percentage=0.0 overMPH=0")
        } else {
            print(
                String(
                    format: "[OverLimit] speedMPH=%@ percentage=%.4f overMPH=%@",
                    speedText,
                    Double(p),
                    overText
                )
            )
        }
        #endif
    }

    // MARK: - Expandable ADAS status row

    /// DRIVER + ROAD status chips.
    private var adasStatusRow: some View {
        HStack(spacing: 6) {
            ADASStatusItem(
                icon: "person.fill",
                title: "DRIVER",
                status: driverDisplayText,
                tone: driverTone
            )
            ADASStatusItem(
                icon: unifiedRoadDisplay.iconName,
                title: "ROAD",
                status: roadDisplayText,
                tone: roadTone
            )
        }
    }

    private var onDeviceFooter: some View {
        HStack(spacing: 4) {
            Image(systemName: "lock.fill")
                .font(.caption2)
            Text("On-device AI")
                .font(.caption2.weight(.medium))
        }
        .foregroundStyle(.secondary)
        .frame(maxWidth: .infinity)
    }

    // MARK: - Display mapping (UI only)

    /// Single ROAD presentation — both analyzers stay active; only display is prioritized.
    private var unifiedRoadDisplay: UnifiedRoadDisplayState {
        UnifiedRoadDisplayState.resolve(
            modelReady: roadDetector.isModelReady,
            modelUnavailable: roadDetector.state == .modelUnavailable,
            vehicle: roadRiskAnalyzer.state,
            pedestrian: pedestrianRiskAnalyzer.state
        )
    }

    private var driverDisplayText: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "Attentive"
        case .lookingAway:
            return "Watch road"
        case .noFace:
            return "Driver not detected"
        }
    }

    private var driverTone: ADASStatusItem.Tone {
        switch driverMonitor.attentionState {
        case .attentive:
            return .normal
        case .lookingAway:
            return .urgent
        case .noFace:
            return .caution
        }
    }

    private var roadDisplayText: String {
        unifiedRoadDisplay.statusText
    }

    private var roadTone: ADASStatusItem.Tone {
        switch unifiedRoadDisplay {
        case .unavailable:
            return .caution
        case .clear, .vehicleAhead, .pedestrianAhead:
            return .normal
        case .closingVehicle:
            return .caution
        case .rapidClosing, .pedestrianClose:
            return .urgent
        }
    }

    /// Priority: Pedestrian Close! > Rapid closing > lane drift > looking away >
    /// major speeding > wake up / no face. Closing vehicle has no dedicated banner.
    private var warningBanner: (title: String, style: WarningBannerView.Style)? {
        if roadDetector.isModelReady {
            if pedestrianRiskAnalyzer.state == .close {
                return ("PEDESTRIAN CLOSE!", .critical)
            }
            if roadRiskAnalyzer.state == .high {
                return ("VEHICLE CLOSING", .critical)
            }
        }

        switch laneDetector.result.state {
        case .driftingLeft:
            return ("LANE DRIFT LEFT", .urgent)
        case .driftingRight:
            return ("LANE DRIFT RIGHT", .urgent)
        case .unavailable, .tracking:
            break
        }

        if driverMonitor.attentionState == .lookingAway {
            return ("WATCH THE ROAD", .urgent)
        }

        if speedingState == .major {
            return ("SLOW DOWN", .urgent)
        }

        if drowsinessRemote.isWakeUpAlertActive {
            return (wakeUpDebugBannerTitle, .caution)
        }

        if driverMonitor.attentionState == .noFace {
            return ("Driver not detected", .caution)
        }

        return nil
    }

    private func logUnifiedRoadDisplayIfNeeded() {
        #if DEBUG
        let text = unifiedRoadDisplay.statusText
        guard text != lastRoadDisplayLog else { return }
        if !lastRoadDisplayLog.isEmpty {
            print("[RoadDisplay] \(lastRoadDisplayLog) -> \(text)")
        }
        lastRoadDisplayLog = text
        #endif
    }

    // MARK: - MultiCam lifecycle (unchanged behavior)

    private func startMultiCamIfNeeded() {
        guard !multiCamOwner.hasStarted else { return }
        multiCamOwner.hasStarted = true
        multiCamOwner.errorMessage = nil

        // External processing only — do NOT call legacy CameraManager start APIs.
        driverMonitor.drowsinessCoordinator = drowsinessRemote
        driverMonitor.beginExternalFrameProcessing()
        roadDetector.beginExternalFrameProcessing()
        laneDetector.beginExternalFrameProcessing()

        let manager = multiCamOwner.manager
        manager.onFrontFrame = { [weak driverMonitor] pixelBuffer in
            driverMonitor?.processExternalFrame(pixelBuffer)
        }
        manager.onRearFrame = { [weak roadDetector, weak laneDetector] pixelBuffer in
            roadDetector?.processExternalFrame(pixelBuffer)
            laneDetector?.processExternalFrame(pixelBuffer)
        }

        manager.requestAccessAndStart { result in
            switch result {
            case .success:
                multiCamOwner.isActive = true
                multiCamOwner.errorMessage = nil
            case .failure(let error):
                multiCamOwner.isActive = false
                multiCamOwner.errorMessage = error.localizedDescription
                driverMonitor.endExternalFrameProcessing()
                roadDetector.endExternalFrameProcessing()
                laneDetector.endExternalFrameProcessing()
                roadRiskAnalyzer.reset()
                pedestrianRiskAnalyzer.reset()
                // Allow a later onAppear retry after a failed start.
                multiCamOwner.hasStarted = false
            }
        }
    }

    private func stopMultiCam() {
        guard multiCamOwner.hasStarted || multiCamOwner.isActive else { return }

        multiCamOwner.manager.onFrontFrame = nil
        multiCamOwner.manager.onRearFrame = nil
        driverMonitor.endExternalFrameProcessing()
        roadDetector.endExternalFrameProcessing()
        laneDetector.endExternalFrameProcessing()
        roadRiskAnalyzer.reset()
        pedestrianRiskAnalyzer.reset()

        multiCamOwner.manager.stop {
            multiCamOwner.isActive = false
            multiCamOwner.hasStarted = false
        }
    }
}

/// Holds the shared MultiCamManager and publishes DriveView status.
@MainActor
final class MultiCamSessionOwner: ObservableObject {
    let manager = MultiCamManager()

    @Published var isActive = false
    @Published var errorMessage: String?
    var hasStarted = false
}

#Preview {
    DriveView()
}
