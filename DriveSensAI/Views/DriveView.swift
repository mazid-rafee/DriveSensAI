//
//  DriveView.swift
//  DriveSensAI
//

import SwiftUI

struct DriveView: View {
    @StateObject private var driverMonitor = DriverMonitor()

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
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemBackground))
        .onAppear {
            driverMonitor.start()
        }
        .onDisappear {
            driverMonitor.stop()
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
