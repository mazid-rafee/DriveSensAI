//
//  RouteRiskScorer.swift
//  DriveSensAI
//

import Foundation
import UIKit

/// Relative safety ranking among currently scored preview routes.
/// Comparative only — not absolute safety claims.
enum RouteSafetyTier: String, Equatable, Sendable {
    case safest
    case medium
    case unsafest
}

/// Styling for safety-colored route polylines and cards.
enum RouteSafetyStyle {
    static let safestColor = UIColor(red: 52 / 255, green: 168 / 255, blue: 83 / 255, alpha: 1)   // #34A853
    static let mediumColor = UIColor(red: 251 / 255, green: 188 / 255, blue: 4 / 255, alpha: 1)    // #FBBC04
    static let unsafestColor = UIColor(red: 234 / 255, green: 67 / 255, blue: 53 / 255, alpha: 1)  // #EA4335
    /// Insufficient safety information (e.g. too many OOV H3 cells) or unscored routes.
    static let unscoredColor = UIColor.systemGray

    static let selectedOpacity: CGFloat = 1.0
    static let unselectedOpacity: CGFloat = 0.35

    static let selectedStrokeWidth: CGFloat = 7
    static let unselectedStrokeWidth: CGFloat = 5

    static let selectedZIndex: Int32 = 60
    static let unselectedZIndex: Int32 = 40

    /// OOV fraction above this → route has insufficient safety information.
    static let insufficientOOVFractionThreshold = 0.97

    static func baseColor(for tier: RouteSafetyTier) -> UIColor {
        switch tier {
        case .safest: return safestColor
        case .medium: return mediumColor
        case .unsafest: return unsafestColor
        }
    }

    static func baseColor(for tier: RouteSafetyTier?) -> UIColor {
        guard let tier else { return unscoredColor }
        return baseColor(for: tier)
    }

    static func strokeColor(tier: RouteSafetyTier?, isSelected: Bool) -> UIColor {
        let opacity = isSelected ? selectedOpacity : unselectedOpacity
        return baseColor(for: tier).withAlphaComponent(opacity)
    }

    static func strokeWidth(isSelected: Bool) -> CGFloat {
        isSelected ? selectedStrokeWidth : unselectedStrokeWidth
    }

    static func zIndex(isSelected: Bool) -> Int32 {
        isSelected ? selectedZIndex : unselectedZIndex
    }
}

/// `safetyScore(route) = severity_weighted_sum` for the active (departure) 3-hour bin.
/// Lower sum = safer. Routes with OOV fraction &gt; 97% are marked insufficient (gray, score 0).
enum RouteRiskScorer {
    /// Applies active-bin model sums → safety scores and relative tiers.
    /// Does not reorder the array (Google / extraction order is preserved).
    static func applyingPredictionScores(
        to routes: [ComputedRoute],
        response: RoutePredictionResponse
    ) -> [ComputedRoute] {
        guard !routes.isEmpty else { return [] }

        let predictionByID = Dictionary(
            uniqueKeysWithValues: response.routes.map { ($0.routeID, $0) }
        )

        var scored = routes.map { route -> ComputedRoute in
            var copy = route
            copy.safetyTier = nil
            copy.hasInsufficientSafetyInfo = false

            guard let prediction = predictionByID[route.id] else {
                copy.safetyScore = nil
                return copy
            }

            if isInsufficientSafetyInfo(prediction) {
                copy.hasInsufficientSafetyInfo = true
                copy.safetyScore = 0
                copy.safetyTier = nil
                return copy
            }

            let sum = prediction.predictionSummary.sum
            if sum.isFinite {
                copy.safetyScore = sum
            } else {
                copy.safetyScore = nil
            }
            return copy
        }

        // Tiers only among routes with enough in-vocabulary coverage.
        let informativeRankedIDs = rankedSafestFirst(scored)
            .filter { !$0.hasInsufficientSafetyInfo && $0.safetyScore != nil }
            .map(\.id)
        let tiersByID = tiers(forRankedIDs: informativeRankedIDs)
        for index in scored.indices {
            if scored[index].hasInsufficientSafetyInfo {
                scored[index].safetyTier = nil
            } else {
                scored[index].safetyTier = tiersByID[scored[index].id]
            }
        }
        return scored
    }

    /// True when OOV cells are more than 97% of traversed H3 cells (or no cells).
    static func isInsufficientSafetyInfo(_ prediction: RoutePrediction) -> Bool {
        let total = prediction.cellCount
        guard total > 0 else { return true }
        let oovFraction = Double(prediction.outOfVocabularyCount) / Double(total)
        return oovFraction > RouteSafetyStyle.insufficientOOVFractionThreshold
    }

    /// Ranked safest-first: lowest sum among informative routes, then insufficient/unscored last.
    static func rankedSafestFirst(_ routes: [ComputedRoute]) -> [ComputedRoute] {
        routes.sorted(by: saferThan)
    }

    /// Safest informative route ID (lowest sum). Ignores insufficient-info routes.
    static func safestRouteID(in routes: [ComputedRoute]) -> String? {
        rankedSafestFirst(routes)
            .first(where: { !$0.hasInsufficientSafetyInfo && $0.safetyScore != nil })?
            .id
    }

    /// Lower safety score (sum) ranks safer among informative routes.
    /// Insufficient-info and missing scores sort last; ties by response index.
    static func saferThan(_ lhs: ComputedRoute, _ rhs: ComputedRoute) -> Bool {
        let leftOK = !lhs.hasInsufficientSafetyInfo && lhs.safetyScore != nil
        let rightOK = !rhs.hasInsufficientSafetyInfo && rhs.safetyScore != nil
        if leftOK != rightOK {
            return leftOK && !rightOK
        }
        if !leftOK && !rightOK {
            return lhs.responseIndex < rhs.responseIndex
        }
        let leftScore = lhs.safetyScore ?? .greatestFiniteMagnitude
        let rightScore = rhs.safetyScore ?? .greatestFiniteMagnitude
        if leftScore != rightScore {
            return leftScore < rightScore
        }
        return lhs.responseIndex < rhs.responseIndex
    }

    static func tiers(forRankedIDs rankedIDs: [String]) -> [String: RouteSafetyTier] {
        let count = rankedIDs.count
        guard count > 0 else { return [:] }

        var result: [String: RouteSafetyTier] = [:]
        switch count {
        case 1:
            result[rankedIDs[0]] = .safest
        case 2:
            result[rankedIDs[0]] = .safest
            result[rankedIDs[1]] = .unsafest
        case 3:
            result[rankedIDs[0]] = .safest
            result[rankedIDs[1]] = .medium
            result[rankedIDs[2]] = .unsafest
        default:
            result[rankedIDs[0]] = .safest
            result[rankedIDs[count - 1]] = .unsafest
            for id in rankedIDs.dropFirst().dropLast() {
                result[id] = .medium
            }
        }
        return result
    }

    #if DEBUG
    static func logScores(_ routes: [ComputedRoute], selectedRouteID: String?) {
        let ordered = rankedSafestFirst(routes)
        for route in ordered {
            let score = route.safetyScore.map { String(format: "%.6g", $0) } ?? "nil"
            let tier = route.safetyTier?.rawValue
                ?? (route.hasInsufficientSafetyInfo ? "insufficient" : "nil")
            let selected = route.id == selectedRouteID
            print("[ROUTE_SAFETY_SCORE] \(route.id) safety_sum=\(score) tier=\(tier) selected=\(selected)")
        }
    }
    #endif
}
