import Foundation

struct PedalCommand {
    var throttle: Double
    var brake: Double
}

enum LongitudinalModel {
    /// Cruise target in m/s. Curves, mass, and power change it per car.
    static func targetSpeed(car: CarModel, pathCurvature: Double) -> Double {
        let comfort = max(2.2, 4.2 - car.curbWeightKG / 1800)
        let curveCap: Double
        if abs(pathCurvature) < 0.002 {
            curveCap = 16
        } else {
            curveCap = sqrt(comfort / abs(pathCurvature))
        }
        let hpPerKg = car.horsepower / max(400, car.curbWeightKG)
        let powerCap = 8 + hpPerKg * 40
        return min(13.5, curveCap, powerCap)
    }

    static func pedals(car: CarModel, speed: Double, target: Double) -> PedalCommand {
        let error = target - speed
        if error >= 0 {
            let hpPerTonne = car.horsepower / (car.curbWeightKG / 1000)
            let authority = min(1, hpPerTonne / 220)
            let throttle = min(1, (error / 3.5) * (0.35 + 0.65 * authority))
            return PedalCommand(throttle: throttle, brake: 0)
        }
        return PedalCommand(throttle: 0, brake: min(1, -error / 3.5))
    }

    /// Meters per second squared. Torque, tire radius, power, mass, and drag all enter.
    static func acceleration(car: CarModel, speed: Double, throttle: Double, brake: Double) -> Double {
        let mass = max(400, car.curbWeightKG)
        let throttleClamped = min(1, max(0, throttle))
        let watts = car.horsepower * 745.7 * throttleClamped
        let torqueForce = (car.torqueNm * 9.0 * throttleClamped) / max(0.2, car.tireRadiusM)
        let powerForce = watts / max(speed, 3)
        let tractionCap = mass * 6.0
        let traction = min(tractionCap, torqueForce, powerForce)
        let drag = 0.5 * 1.225 * car.dragCoefficient * car.frontalAreaM2 * speed * speed
        let roll = 0.013 * mass * 9.81
        let drive: Double
        if throttleClamped > 0.02 {
            drive = (traction - drag - roll) / mass
        } else {
            drive = -(drag + roll) / mass
        }
        return drive - min(1, max(0, brake)) * 6.5
    }
}
