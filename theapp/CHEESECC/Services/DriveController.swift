import Combine
import Foundation
import SwiftData

@MainActor
final class DriveController: ObservableObject {
    @Published var route: [LatLon] = []
    @Published var destinationName = ""
    @Published var destination: LatLon?
    @Published var running = false
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

    var context: ModelContext?
    var locationSource: LocationService?

    private var path: RoutePath?
    private var tick: Task<Void, Never>?
    private var sequence: UInt64 = 0
    private var session: DriveSession?
    private var sinceCommand = 0.0
    private var generation = 0
    private var commands: [DriveCommand] = []
    private var offRouteSince: Date?
    private var rerouteNotBefore = Date.distantPast
    private var rerouteGeneration = 0
    private var rerouting = false

    private static let offRouteMeters = 60.0
    private static let offRouteDwell: TimeInterval = 1.5
    private static let rerouteCooldown: TimeInterval = 6

    func setRoute(_ coordinates: [LatLon], name: String, destination: LatLon) {
        stop(updateStatus: nil)
        install(coordinates, name: name, destination: destination, snapToFix: false)
        status = "Route ready. \(Units.meters(routeLengthM)) to \(name)."
    }

    /// Swaps the polyline without ending a live session. Snaps progress to the phone.
    func replaceRoute(_ coordinates: [LatLon], name: String, destination: LatLon) {
        guard running else {
            setRoute(coordinates, name: name, destination: destination)
            return
        }
        cancelReroute()
        rerouteNotBefore = Date().addingTimeInterval(Self.rerouteCooldown)
        install(coordinates, name: name, destination: destination, snapToFix: true)
        status = "Following GPS."
    }

    func clearRoute() {
        let wasRunning = running
        stop(updateStatus: nil)
        route = []
        destinationName = ""
        destination = nil
        path = nil
        routeLengthM = 0
        distanceAlong = 0
        speedMps = 0
        lookahead = nil
        commands = []
        commandLog = ""
        sinceCommand = 0
        if let fix = locationSource?.coordinate {
            position = fix
        }
        status = wasRunning ? "Stopped." : "Search a destination, then tap Go."
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
        guard let fix = locationSource?.coordinate else {
            status = "Waiting for a GPS lock."
            return
        }

        generation += 1
        let token = generation
        tick?.cancel()
        activeCar = selectedCar
        sinceCommand = 0
        offRouteSince = nil

        let snap = path.closest(to: fix)
        distanceAlong = snap.along
        position = fix
        if let speed = locationSource?.speedMps, speed >= 0 {
            speedMps = speed
        }
        headingDeg = heading(along: snap.along, path: path)

        let record = DriveSession(carId: selectedCar.id, carName: selectedCar.displayName)
        context?.insert(record)
        try? context?.save()
        session = record
        running = true
        status = "Following GPS."

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
        cancelReroute()
        running = false
        tick?.cancel()
        tick = nil
        session = nil
        if let updateStatus {
            status = updateStatus
        }
    }

    private func step(wallDT: Double) {
        guard running, let path else { return }
        applyLiveFix(path: path)
        guard running, let livePath = self.path, let car = activeCar else { return }

        let solved = VehicleCommandSolver.solve(
            path: livePath,
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

        sinceCommand += wallDT
        if sinceCommand + 0.0001 >= solved.sample.lookaheadS {
            sinceCommand = 0
            emit(
                car: car,
                solved: solved,
                from: solvedFrom,
                source: usingDriver ? "driver" : "planner"
            )
        }

        if distanceAlong >= livePath.totalM - 1.5 {
            stop(updateStatus: "Arrived at \(destinationName).")
        }
    }

    private func applyLiveFix(path: RoutePath) {
        guard let fix = locationSource?.coordinate else {
            offRouteSince = nil
            if !rerouting {
                setStatus("GPS dropped.")
            }
            return
        }
        let snap = path.closest(to: fix)
        distanceAlong = snap.along
        position = fix
        if let speed = locationSource?.speedMps, speed >= 0 {
            speedMps = speed
        }
        headingDeg = heading(along: snap.along, path: path)

        let now = Date()
        if snap.offsetM > Self.offRouteMeters {
            if offRouteSince == nil {
                offRouteSince = now
            }
            let stayed = now.timeIntervalSince(offRouteSince ?? now) >= Self.offRouteDwell
            if stayed, now >= rerouteNotBefore, !rerouting {
                beginReroute(from: fix)
            }
        } else {
            offRouteSince = nil
            if !rerouting {
                setStatus("Following GPS.")
            }
        }
    }

    private func beginReroute(from fix: LatLon) {
        guard running, !rerouting, let destination else { return }
        rerouting = true
        rerouteGeneration += 1
        let token = rerouteGeneration
        let name = destinationName
        rerouteNotBefore = Date().addingTimeInterval(Self.rerouteCooldown)
        offRouteSince = nil
        setStatus("Recalculating…")

        Task { @MainActor [weak self] in
            do {
                let line = try await RouteService.driveRoute(from: fix, to: destination)
                guard let self, self.rerouteGeneration == token, self.running else { return }
                self.replaceRoute(line, name: name, destination: destination)
            } catch {
                guard let self, self.rerouteGeneration == token, self.running else { return }
                self.rerouting = false
                self.setStatus("Following GPS.")
            }
        }
    }

    private func install(
        _ coordinates: [LatLon],
        name: String,
        destination: LatLon,
        snapToFix: Bool
    ) {
        route = coordinates
        destinationName = name
        self.destination = destination
        let built = RoutePath(coordinates: coordinates)
        path = built
        routeLengthM = built.totalM

        if snapToFix, coordinates.count >= 2 {
            if let fix = locationSource?.coordinate {
                let snap = built.closest(to: fix)
                distanceAlong = snap.along
                position = fix
                if let speed = locationSource?.speedMps, speed >= 0 {
                    speedMps = speed
                }
                headingDeg = heading(along: snap.along, path: built)
            }
        } else {
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
        }

        if coordinates.isEmpty {
            lookahead = nil
        } else {
            lookahead = built.point(atDistance: min(12, built.totalM)).coordinate
        }
    }

    private func heading(along: Double, path: RoutePath) -> Double {
        if speedMps > 1.5, let course = locationSource?.courseDeg, course >= 0 {
            return course
        }
        return path.point(atDistance: along).bearingDeg
    }

    private func cancelReroute() {
        rerouteGeneration += 1
        rerouting = false
        offRouteSince = nil
        rerouteNotBefore = .distantPast
    }

    private func setStatus(_ text: String) {
        if status != text {
            status = text
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
