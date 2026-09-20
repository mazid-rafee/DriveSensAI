//
//  RouteOptionCard.swift
//  DriveSensAI
//

import SwiftUI

/// Compact selectable card for one mock-scored preview route.
struct RouteOptionCard: View {
    let displayRank: Int
    let route: ComputedRoute
    let isSelected: Bool
    let onSelect: () -> Void

    private var safetyColor: Color {
        Color(uiColor: MockRouteSafetyStyle.baseColor(for: route.safetyTier ?? .medium))
    }

    private var accessibilitySafetyPhrase: String {
        switch route.safetyTier {
        case .safest: return "safest"
        case .medium: return "medium safety"
        case .unsafest: return "unsafest"
        case .none: return "medium safety"
        }
    }

    var body: some View {
        Button(action: onSelect) {
            VStack(spacing: 2) {
                Text(route.durationText)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.70)
                    .frame(maxWidth: .infinity)

                Text(route.distanceMilesText)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.white.opacity(0.72))
                    .lineLimit(1)
                    .minimumScaleFactor(0.70)
                    .frame(maxWidth: .infinity)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, minHeight: 44)
            .background {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.black.opacity(isSelected ? 0.92 : 0.78))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(safetyColor.opacity(isSelected ? 0.18 : 0.0))
                    }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .strokeBorder(
                        isSelected ? safetyColor : safetyColor.opacity(0.55),
                        lineWidth: isSelected ? 3 : 2.5
                    )
            }
            .shadow(
                color: isSelected ? safetyColor.opacity(0.35) : .clear,
                radius: isSelected ? 4 : 0,
                y: isSelected ? 1 : 0
            )
            .opacity(1.0)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .animation(.easeInOut(duration: 0.15), value: isSelected)
        .accessibilityElement(children: .ignore)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityLabel(
            "Route \(displayRank), \(accessibilitySafetyPhrase), \(route.durationText), \(route.distanceMilesText)"
        )
    }
}
