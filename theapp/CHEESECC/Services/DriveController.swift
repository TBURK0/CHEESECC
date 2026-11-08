import Combine
import Foundation
import SwiftData

@MainActor
final class DriveController: ObservableObject {
    @Published var route: [LatLon] = []
    @Published var destinationName = ""
    @Published var running = false
    @Published var replay = true
    @Published var status = "Search a destination, then tap Go."
    @Published var speedMps = 0.0
    @Published var headingDeg = 0.0
    @Published var position = DemoOrigin.cincinnati
    @Published var lookahead: LatLon?
    @Published var lookaheadS = 0.5
    @Published var steerWheelDeg = 0.0
    @Published var plannerSteerDeg = 0.0
    @Published var roadWheelDeg = 0.0
    @Published var throttle = 0.0
    @Published var brake = 0.0
    @Published var curvature = 0.0
    @Published var targetSpeedMps = 0.0
    @Published var manualSteer: Double?
    @Published var commandLog = ""
    @Published var distanceAlong = 0.0
    @Published var routeLengthM = 0.0
    @Published var selectedCar: CarModel?
    @Published private(set) var activeCar: CarModel?
    @Published var replayRate = 5.0

    var context: ModelContext?
    var locationSource: LocationService?

    private var path: RoutePath?
    private var tick: Task<Void, Never>?
    private var sequence: UInt64 = 0
    private var session: DriveSession?
    private var sinceCommand = 0.0
    private var generation = 0
    private var commands: [DriveCommand] = []

    func setRoute(_ coordinates: [LatLon], name: String) {
        stop(updateStatus: nil)
        route = coordinates
        destinationName = name
        let built = RoutePath(coordinates: coordinates)
        path = built
        routeLengthM = built.totalM
        distanceAlong = 0
        speedMps = 0
        commands = []
        commandLog = ""
        sinceCommand = 0
        if let first = coordinates.first {
            position = first
        }
        if coordinates.count >= 2 {
            headingDeg = GeoMath.bearingDeg(from: coordinates[0], to: coordinates[1])
        }
        lookahead = built.point(atDistance: 12).coordinate
        status = "Route ready. \(Units.meters(built.totalM)) to \(name)."
    }

    func start() {
        guard let path, path.coordinates.count >= 2 else {
            status = "No route yet."
            return
        }
        guard let selectedCar else {
            status = "Pick a car in Garage."
            return
        }
        if !replay && locationSource?.coordinate == nil {
            status = "Waiting for a GPS lock. Turn on Replay indoors."
            return
        }

        generation += 1
        let token = generation
        tick?.cancel()
        activeCar = selectedCar
        sinceCommand = 0

        if replay {
            distanceAlong = 0
            speedMps = 0
            position = path.coordinates[0]
            headingDeg = GeoMath.bearingDeg(from: path.coordinates[0], to: path.coordinates[1])
        } else if let fix = locationSource?.coordinate {
            let snap = path.closest(to: fix)
            distanceAlong = snap.along
            position = fix
            if let speed = locationSource?.speedMps, speed >= 0 {
                speedMps = speed
            }
            headingDeg = path.point(atDistance: snap.along).bearingDeg
        }

        let record = DriveSession(carId: selectedCar.id, carName: selectedCar.displayName)
        context?.insert(record)
        try? context?.save()
        session = record
        running = true
        status = replay ? "Replay running." : "Following GPS."

        tick = Task { @MainActor [weak self] in
            var last = Date()
            while !Task.isCancelled {
                guard let self else { return }
                guard self.running, self.generation == token else { return }
                let now = Date()
                let wall = min(0.25, max(0, now.timeIntervalSince(last)))
                last = now
                self.step(wallDT: wall)
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
    }

    func stop(updateStatus: String? = "Stopped.") {
        generation += 1
        running = false
        tick?.cancel()
        tick = nil
        session = nil
        if let updateStatus {
            status = updateStatus
        }
    }

    private func step(wallDT: Double) {
        guard running, let path, let car = activeCar else { return }
        let dt = replay ? wallDT * replayRate : wallDT

        if !replay {
            applyLiveFix(path: path)
        }

        let solved = VehicleCommandSolver.solve(
            path: path,
            distanceAlong: distanceAlong,
            position: position,
            headingDeg: headingDeg,
            speedMps: speedMps,
            car: car
        )
        lookahead = solved.sample.target
        lookaheadS = solved.sample.lookaheadS
        plannerSteerDeg = solved.steer.steerWheelDeg
        roadWheelDeg = solved.steer.roadWheelDeg
        curvature = solved.steer.curvature
        targetSpeedMps = solved.targetSpeedMps
        throttle = solved.pedals.throttle
        brake = solved.pedals.brake
        let usingDriver = manualSteer != nil
        steerWheelDeg = manualSteer ?? solved.steer.steerWheelDeg
        let solvedFrom = position

        sinceCommand += dt
        if sinceCommand + 0.0001 >= solved.sample.lookaheadS {
            sinceCommand = 0
            emit(
                car: car,
                solved: solved,
                from: solvedFrom,
                source: usingDriver ? "driver" : "planner"
            )
        }

        if replay {
            let accel = LongitudinalModel.acceleration(
                car: car,
                speed: speedMps,
                throttle: throttle,
                brake: brake
            )
            speedMps = max(0, speedMps + accel * dt)
            distanceAlong = min(path.totalM, distanceAlong + speedMps * dt)
            let pose = path.point(atDistance: distanceAlong)
            position = pose.coordinate
            headingDeg = pose.bearingDeg
            status = "Replay \(Units.mph(speedMps))."
        }

        if distanceAlong >= path.totalM - 1.5 {
            stop(updateStatus: "Arrived at \(destinationName).")
        }
    }

    private func applyLiveFix(path: RoutePath) {
        guard let fix = locationSource?.coordinate else {
            status = "GPS dropped. Replay is still available."
            return
        }
        let snap = path.closest(to: fix)
        distanceAlong = snap.along
        position = fix
        if let speed = locationSource?.speedMps, speed >= 0 {
            speedMps = speed
        }
        if speedMps > 1.5, let course = locationSource?.courseDeg, course >= 0 {
            headingDeg = course
        } else {
            headingDeg = path.point(atDistance: snap.along).bearingDeg
        }
        if snap.offsetM > 60 {
            status = "Off the route by \(Units.meters(snap.offsetM))."
        } else {
            status = "Following GPS."
        }
    }

    private func emit(car: CarModel, solved: SolvedCommand, from: LatLon, source: String) {
        sequence += 1
        let command = DriveCommand(
            id: sequence,
            lookaheadS: round(solved.sample.lookaheadS, places: 100),
            latitude: round(solved.sample.target.latitude, places: 1_000_000),
            longitude: round(solved.sample.target.longitude, places: 1_000_000),
            fromLatitude: round(from.latitude, places: 1_000_000),
            fromLongitude: round(from.longitude, places: 1_000_000),
            bearingDeg: round(headingDeg, places: 10),
            speedMps: round(speedMps, places: 100),
            targetSpeedMps: round(solved.targetSpeedMps, places: 100),
            steerWheelDeg: round(steerWheelDeg, places: 10),
            throttle: round(throttle, places: 100),
            brake: round(brake, places: 100),
            curvature: round(curvature, places: 10_000),
            carId: car.id,
            source: source
        )
        commands.append(command)
        if commands.count > 40 {
            commands.removeFirst(commands.count - 40)
        }
        commandLog = commands.suffix(4).map(\.line).joined(separator: "\n")

        guard let session, let context else { return }
        let row = SteerSample(
            time: Date(),
            latitude: from.latitude,
            longitude: from.longitude,
            speedMps: speedMps,
            headingDeg: headingDeg,
            curvature: curvature,
            steerWheelDeg: steerWheelDeg,
            throttle: throttle,
            brake: brake,
            lookaheadS: solved.sample.lookaheadS,
            source: source,
            carId: car.id
        )
        row.session = session
        context.insert(row)
        try? context.save()
    }

    private func round(_ value: Double, places: Double) -> Double {
        (value * places).rounded() / places
    }
}
