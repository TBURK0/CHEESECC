import Foundation

struct SolvedCommand {
    var sample: RouteSample
    var steer: SteerSolution
    var targetSpeedMps: Double
    var pedals: PedalCommand
}

enum VehicleCommandSolver {
    /// One control tick: where the car should be next, and the wheel and pedals that point it there.
    static func solve(
        path: RoutePath,
        distanceAlong: Double,
        position: LatLon,
        headingDeg: Double,
        speedMps: Double,
        car: CarModel
    ) -> SolvedCommand {
        let sample = RouteDiscretizer.sample(
            path: path,
            distanceAlong: distanceAlong,
            position: position,
            speedMps: speedMps
        )
        let headingError = GeoMath.wrapDegrees(sample.bearingToTargetDeg - headingDeg) * .pi / 180
        let lookahead = max(sample.lookaheadM, GeoMath.distanceM(position, sample.target))
        let steer = BicycleModel.steer(
            headingErrorRad: headingError,
            lookaheadM: lookahead,
            car: car
        )
        let targetSpeed = LongitudinalModel.targetSpeed(car: car, pathCurvature: sample.pathCurvature)
        let pedals = LongitudinalModel.pedals(car: car, speed: speedMps, target: targetSpeed)
        return SolvedCommand(
            sample: sample,
            steer: steer,
            targetSpeedMps: targetSpeed,
            pedals: pedals
        )
    }
}
