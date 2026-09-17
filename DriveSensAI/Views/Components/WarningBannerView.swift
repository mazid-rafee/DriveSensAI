//
//  WarningBannerView.swift
//  DriveSensAI
//

import SwiftUI

struct WarningBannerView: View {
    enum Style {
        case urgent
        case caution
    }

    let title: String
    let style: Style

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: style == .urgent
                  ? "exclamationmark.triangle.fill"
                  : "person.crop.circle.badge.exclamationmark")
                .font(.title3.weight(.semibold))

            Text(title)
                .font(.headline.weight(.bold))
                .tracking(0.6)
        }
        .foregroundStyle(style == .urgent ? Color.black : Color.primary)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(style == .urgent ? Color.orange : Color.white.opacity(0.12))
        )
        .accessibilityAddTraits(.isHeader)
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
