import Foundation

struct RouteSample {
    var target: LatLon
    var tangentBearingDeg: Double
    var bearingToTargetDeg: Double
    var lookaheadM: Double
    var lookaheadS: Double
    var pathCurvature: Double
}

enum RouteDiscretizer {
    static func sample(
        path: RoutePath,
        distanceAlong: Double,
        position: LatLon,
        speedMps: Double
    ) -> RouteSample {
        let horizon = LookaheadScheduler.horizon(speedMps: speedMps)
        let lookahead = LookaheadScheduler.distance(speedMps: speedMps)
        let ahead = path.point(atDistance: distanceAlong + lookahead)
        return RouteSample(
            target: ahead.coordinate,
            tangentBearingDeg: ahead.bearingDeg,
            bearingToTargetDeg: GeoMath.bearingDeg(from: position, to: ahead.coordinate),
            lookaheadM: lookahead,
            lookaheadS: horizon,
            pathCurvature: path.curvature(atDistance: distanceAlong)
        )
    }
}
