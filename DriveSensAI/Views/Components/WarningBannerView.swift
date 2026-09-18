//
//  WarningBannerView.swift
//  DriveSensAI
//

import SwiftUI

struct WarningBannerView: View {
    enum Style {
        case urgent
        case caution
        case critical
    }

    let title: String
    let style: Style

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: iconName)
                .font(.title3.weight(.semibold))

            Text(title)
                .font(.headline.weight(.bold))
                .tracking(0.6)
        }
        .foregroundStyle(foreground)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(background)
        )
        .accessibilityAddTraits(.isHeader)
    }

    private var iconName: String {
        switch style {
        case .urgent, .critical:
            return "exclamationmark.triangle.fill"
        case .caution:
            return "person.crop.circle.badge.exclamationmark"
        }
    }

    private var foreground: Color {
        switch style {
        case .urgent, .critical:
            return .black
        case .caution:
            return .primary
        }
    }

    private var background: Color {
        switch style {
        case .urgent:
            return .orange
        case .critical:
            return .red
        case .caution:
            return Color.white.opacity(0.12)
        }
    }
}

#Preview {
    VStack(spacing: 12) {
        WarningBannerView(title: "WATCH THE ROAD", style: .urgent)
        WarningBannerView(title: "Driver not detected", style: .caution)
    }
    .padding()
    .preferredColorScheme(.dark)
    .background(Color.black)
}
