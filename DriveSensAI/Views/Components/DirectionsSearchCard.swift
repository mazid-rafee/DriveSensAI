//
//  DirectionsSearchCard.swift
//  DriveSensAI
//

import SwiftUI

enum DirectionsSearchField: Hashable {
    case source
    case destination
}

/// Google Maps–style source / destination card with editable TextFields.
struct DirectionsSearchCard: View {
    @Binding var sourceText: String
    @Binding var destinationText: String
    var focusedField: FocusState<DirectionsSearchField?>.Binding

    var onSourceTextChange: (String) -> Void
    var onDestinationTextChange: (String) -> Void
    var onClearSource: () -> Void
    var onClearDestination: () -> Void
    var onSwap: () -> Void

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            VStack(spacing: 0) {
                fieldRow(
                    field: .source,
                    text: $sourceText,
                    placeholder: "Choose starting point",
                    indicatorColor: .blue,
                    accessibilityLabel: "Starting point",
                    onClear: onClearSource
                )

                Divider()
                    .padding(.leading, 28)

                fieldRow(
                    field: .destination,
                    text: $destinationText,
                    placeholder: "Choose destination",
                    indicatorColor: .red,
                    accessibilityLabel: "Destination",
                    onClear: onClearDestination
                )
            }

            Button(action: onSwap) {
                Image(systemName: "arrow.up.arrow.down")
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 44, minHeight: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Swap starting point and destination")
        }
        .padding(.leading, 14)
        .padding(.trailing, 6)
        .padding(.vertical, 6)
        .background {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(.systemBackground))
                .shadow(color: Color.black.opacity(0.18), radius: 10, x: 0, y: 4)
        }
        .accessibilityElement(children: .contain)
        // Observe the bound strings directly so each keystroke is delivered immediately.
        .onChange(of: sourceText) { _, newValue in
            onSourceTextChange(newValue)
        }
        .onChange(of: destinationText) { _, newValue in
            onDestinationTextChange(newValue)
        }
    }

    private func fieldRow(
        field: DirectionsSearchField,
        text: Binding<String>,
        placeholder: String,
        indicatorColor: Color,
        accessibilityLabel: String,
        onClear: @escaping () -> Void
    ) -> some View {
        let isActive = focusedField.wrappedValue == field

        return HStack(spacing: 12) {
            Circle()
                .fill(indicatorColor)
                .frame(width: 10, height: 10)
                .accessibilityHidden(true)

            TextField(placeholder, text: text)
                .font(.body)
                .textInputAutocapitalization(.words)
                .disableAutocorrection(true)
                .focused(focusedField, equals: field)
                .submitLabel(.search)
                .accessibilityLabel(accessibilityLabel)

            if !text.wrappedValue.isEmpty {
                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Clear \(accessibilityLabel.lowercased())")
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 4)
        .background {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(isActive ? Color.accentColor.opacity(0.12) : Color.clear)
        }
        .contentShape(Rectangle())
        .simultaneousGesture(TapGesture().onEnded {
            focusedField.wrappedValue = field
        })
    }
}
