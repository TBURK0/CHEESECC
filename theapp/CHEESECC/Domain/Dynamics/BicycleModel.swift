import Foundation

struct SteerSolution {
    var steerWheelDeg: Double
    var curvature: Double
    var roadWheelDeg: Double
}

enum BicycleModel {
    static func steer(
        headingErrorRad: Double,
        lookaheadM: Double,
        car: CarModel
    ) -> SteerSolution {
        let curvature = PurePursuit.curvature(
            lookaheadM: lookaheadM,
            headingErrorRad: headingErrorRad
        )
        let roadWheelRad = atan(curvature * car.wheelbaseM)
        let handwheel = roadWheelRad * car.steeringRatio * 180 / .pi
        let clamped = min(car.maxSteerWheelDeg, max(-car.maxSteerWheelDeg, handwheel))
        return SteerSolution(
            steerWheelDeg: clamped,
            curvature: curvature,
            roadWheelDeg: roadWheelRad * 180 / .pi
        )
    }

    /// Small-angle handwheel degrees per 1/m of curvature: L * steeringRatio * 180/π.
    static func theoreticalGain(car: CarModel) -> Double {
        car.wheelbaseM * car.steeringRatio * 180 / .pi
    }
}
