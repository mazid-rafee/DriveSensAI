//
//  RouteRiskScorerTests.swift
//  DriveSensAITests
//

import XCTest
@testable import DriveSensAI

@MainActor
final class RouteRiskScorerTests: XCTestCase {

    func testSafetyScoreUsesSumAndLowerIsSafer() {
        let routes = RouteRiskScorer.applyingPredictionScores(
            to: [makeRoute(index: 0), makeRoute(index: 1), makeRoute(index: 2)],
            response: makeResponse(sums: [
                "route_0": 0.165814,
                "route_1": 0.249322,
                "route_2": 0.217158
            ])
        )
        let ranked = RouteRiskScorer.rankedSafestFirst(routes)
        XCTAssertEqual(ranked.map(\.id), ["route_0", "route_2", "route_1"])
        XCTAssertEqual(routes.first(where: { $0.id == "route_0" })?.safetyScore, 0.165814)
        XCTAssertEqual(ranked[0].safetyTier, .safest)
        XCTAssertEqual(ranked[1].safetyTier, .medium)
        XCTAssertEqual(ranked[2].safetyTier, .unsafest)
    }

    func testHighOOVMarksInsufficientGrayWithZeroScore() {
        let routes = RouteRiskScorer.applyingPredictionScores(
            to: [makeRoute(index: 0), makeRoute(index: 1)],
            response: RoutePredictionResponse(
                requestID: "test",
                modelVersion: "best.pt",
                routes: [
                    RoutePrediction(
                        routeID: "route_0",
                        cellCount: 100,
                        scoredCellCount: 2,
                        outOfVocabularyCount: 98, // 98% > 97%
                        activeHourBinStart: 12,
                        predictionSummary: RoutePredictionSummary(mean: 0.01, maximum: 0.02, sum: 0.02),
                        timeBinScores: [],
                        cells: []
                    ),
                    RoutePrediction(
                        routeID: "route_1",
                        cellCount: 50,
                        scoredCellCount: 48,
                        outOfVocabularyCount: 2,
                        activeHourBinStart: 12,
                        predictionSummary: RoutePredictionSummary(mean: 0.01, maximum: 0.02, sum: 0.48),
                        timeBinScores: [],
                        cells: []
                    )
                ]
            )
        )
        let byID = Dictionary(uniqueKeysWithValues: routes.map { ($0.id, $0) })
        XCTAssertTrue(byID["route_0"]!.hasInsufficientSafetyInfo)
        XCTAssertEqual(byID["route_0"]!.safetyScore, 0)
        XCTAssertNil(byID["route_0"]!.safetyTier)
        XCTAssertFalse(byID["route_1"]!.hasInsufficientSafetyInfo)
        XCTAssertEqual(byID["route_1"]!.safetyTier, .safest)
        XCTAssertEqual(RouteRiskScorer.safestRouteID(in: routes), "route_1")
    }

    func testOOVAtExactly97PercentRemainsInformative() {
        let prediction = RoutePrediction(
            routeID: "route_0",
            cellCount: 100,
            scoredCellCount: 3,
            outOfVocabularyCount: 97,
            activeHourBinStart: 12,
            predictionSummary: RoutePredictionSummary(mean: 0.01, maximum: 0.02, sum: 0.03),
            timeBinScores: [],
            cells: []
        )
        XCTAssertFalse(RouteRiskScorer.isInsufficientSafetyInfo(prediction))
    }

    func testTiesResolvedByGoogleResponseOrder() {
        var a = makeRoute(index: 0)
        var b = makeRoute(index: 1)
        a.safetyScore = 0.2
        b.safetyScore = 0.2
        XCTAssertTrue(RouteRiskScorer.saferThan(a, b))
        XCTAssertFalse(RouteRiskScorer.saferThan(b, a))
    }

    func testStyleOpacityContract() {
        XCTAssertEqual(RouteSafetyStyle.selectedOpacity, 1.0)
        XCTAssertEqual(RouteSafetyStyle.unselectedOpacity, 0.35, accuracy: 0.000_1)
        let selected = RouteSafetyStyle.strokeColor(tier: .safest, isSelected: true)
        let unselected = RouteSafetyStyle.strokeColor(tier: .safest, isSelected: false)
        XCTAssertGreaterThan(selected.cgColor.alpha, unselected.cgColor.alpha)
        XCTAssertEqual(
            RouteSafetyStyle.baseColor(for: nil as RouteSafetyTier?),
            RouteSafetyStyle.unscoredColor
        )
    }

    // MARK: - Helpers

    private func makeResponse(sums: [String: Double]) -> RoutePredictionResponse {
        RoutePredictionResponse(
            requestID: "test-request",
            modelVersion: "best.pt",
            routes: sums.keys.sorted().map { routeID in
                let sum = sums[routeID]!
                return RoutePrediction(
                    routeID: routeID,
                    cellCount: 10,
                    scoredCellCount: 8,
                    outOfVocabularyCount: 2,
                    activeHourBinStart: 12,
                    predictionSummary: RoutePredictionSummary(
                        mean: sum / 8,
                        maximum: sum / 4,
                        sum: sum
                    ),
                    timeBinScores: [
                        TimeBinSafetyScore(
                            hourBinStart: 12,
                            severityWeightedSum: sum,
                            maxSeverityWeightedRate: sum / 4,
                            meanPersonRate: 0.01,
                            meanPropertyRate: 0.01,
                            meanSocietyRate: 0.01,
                            meanOtherRate: 0.01,
                            cellCount: 8
                        )
                    ],
                    cells: []
                )
            }
        )
    }

    private func makeRoute(index: Int) -> ComputedRoute {
        ComputedRoute(
            id: "route_\(index)",
            encodedPolyline: "_p~iF~ps|U",
            durationText: "10 min",
            durationSeconds: 600,
            distanceMeters: 1000,
            sourcePlaceID: "src",
            destinationPlaceID: "dst",
            routeLabels: [],
            isDefault: index == 0,
            responseIndex: index,
            steps: [
                ComputedRouteStep(
                    encodedPolyline: "_p~iF~ps|U",
                    distanceMeters: 1000,
                    staticDurationSeconds: 600
                )
            ],
            isExtractionReady: true,
            safetyScore: nil,
            safetyTier: nil,
            hasInsufficientSafetyInfo: false
        )
    }
}
