//
//  DriveView.swift
//  DriveSensAI
//

import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()
    @State private var showRoadDebug = false

    var body: some View {
        VStack(spacing: 24) {
            Text("DriveSensAI")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text(statusTitle)
                .font(.system(size: 40, weight: .bold, design: .rounded))
                .multilineTextAlignment(.center)
                .foregroundStyle(statusColor)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 48)

            if let error = driverMonitor.cameraError {
                Text(error.localizedDescription)
                    .font(.subheadline)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            } else if !driverMonitor.isRunning {
                Text("Starting camera…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            // Temporary Milestone 2 entry point — stops front camera before rear starts.
            Button("Test Road Detection") {
                driverMonitor.stop()
                showRoadDebug = true
            }
            .buttonStyle(.bordered)
            .padding(.bottom, 8)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .onAppear {
            // Restore front camera when returning from road debug (or first launch).
            if !showRoadDebug {
                driverMonitor.start()
            }
        }
        .onDisappear {
            // Leaving DriveView entirely (e.g. app background / teardown).
            if !showRoadDebug {
                driverMonitor.stop()
            }
        }
        .fullScreenCover(isPresented: $showRoadDebug, onDismiss: {
            driverMonitor.start()
        }) {
            RoadDebugView()
        }
    }

    private var statusTitle: String {
        switch driverMonitor.attentionState {
        case .attentive:
            return "ATTENTIVE"
        case .lookingAway:
            return "LOOKING AWAY"
        case .noFace:
            return "NO FACE"
        }
    }

    private var statusColor: Color {
        switch driverMonitor.attentionState {
        case .attentive:
            return .green
        case .lookingAway:
            return .orange
        case .noFace:
            return .red
        }
    }
}

#Preview {
    DriveView()
}
