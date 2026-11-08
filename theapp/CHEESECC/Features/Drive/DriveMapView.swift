import MapKit
import SwiftUI

struct DriveMapView: View {
    @EnvironmentObject private var drive: DriveController
    @EnvironmentObject private var location: LocationService
    @State private var query = ""
    @State private var results: [PlaceResult] = []
    @State private var searching = false
    @State private var handwheel = 0.0
    @State private var holdingWheel = false
    @State private var camera: MapCameraPosition = .region(
        MKCoordinateRegion(
            center: DemoOrigin.cincinnati.coordinate,
            span: MKCoordinateSpan(latitudeDelta: 0.08, longitudeDelta: 0.08)
        )
    )

    private var shownCar: CarModel? {
        drive.activeCar ?? drive.selectedCar
    }

    var body: some View {
        Map(position: $camera) {
            UserAnnotation()
            if drive.route.count > 1 {
                MapPolyline(coordinates: drive.route.map(\.coordinate))
                    .stroke(Theme.amber, lineWidth: 5)
            }
            Annotation("Car", coordinate: drive.position.coordinate, anchor: .center) {
                Image(systemName: "location.north.fill")
                    .font(.title2)
                    .foregroundStyle(Theme.amber)
                    .rotationEffect(.degrees(drive.headingDeg))
            }
            if let next = drive.lookahead {
                Annotation("Next", coordinate: next.coordinate, anchor: .center) {
                    ZStack {
                        Circle().fill(Theme.amber.opacity(0.35))
                        Circle().stroke(Theme.amber, lineWidth: 2)
                    }
                    .frame(width: 16, height: 16)
                }
            }
        }
        .mapStyle(.standard)
        .mapControls {
            MapUserLocationButton()
            MapCompass()
            MapScaleView()
        }
        .safeAreaInset(edge: .top) { searchCard }
        .safeAreaInset(edge: .bottom) { driveDeck }
        .onChange(of: drive.route.count) { _, count in
            guard count > 1 else { return }
            camera = .region(fit(drive.route))
        }
        .onChange(of: location.coordinate?.latitude) { _, _ in
            guard drive.route.isEmpty, let coordinate = location.coordinate else { return }
            camera = .region(
                MKCoordinateRegion(
                    center: coordinate.coordinate,
                    span: MKCoordinateSpan(latitudeDelta: 0.04, longitudeDelta: 0.04)
                )
            )
        }
        .onChange(of: drive.plannerSteerDeg) { _, newValue in
            if !holdingWheel {
                handwheel = newValue
            }
        }
    }

    private var searchCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(shownCar?.displayName ?? "No car")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(location.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack {
                TextField("Destination", text: $query)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.words)
                    .submitLabel(.search)
                    .onSubmit { Task { await search() } }
                Button("Search") { Task { await search() } }
                    .buttonStyle(.borderedProminent)
                    .disabled(searching || query.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            ForEach(results) { place in
                Button(place.name) { Task { await route(to: place) } }
                    .font(.subheadline)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            Text(drive.status)
                .font(.caption)
                .foregroundStyle(Theme.amber)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 12)
        .padding(.top, 4)
    }

    private var driveDeck: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                metric("Speed", Units.mph(drive.speedMps))
                metric("Wheel", String(format: "%+.0f°", drive.steerWheelDeg))
                metric("Next", String(format: "%.2fs", drive.lookaheadS))
                metric("Target", Units.mph(drive.targetSpeedMps))
            }
            if let car = shownCar {
                HStack(alignment: .center, spacing: 12) {
                    ChassisDiagramView(car: car, steerWheelDeg: drive.steerWheelDeg)
                        .frame(width: 108, height: 128)
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Planner \(String(format: "%+.0f°", drive.plannerSteerDeg))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Text(holdingWheel ? "Logging your wheel" : "Planner has the wheel")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Slider(
                            value: $handwheel,
                            in: -car.maxSteerWheelDeg...car.maxSteerWheelDeg,
                            onEditingChanged: { editing in
                                holdingWheel = editing
                                drive.manualSteer = editing ? handwheel : nil
                            }
                        )
                        .onChange(of: handwheel) { _, newValue in
                            if holdingWheel {
                                drive.manualSteer = newValue
                            }
                        }
                        Toggle(isOn: replayBinding) {
                            Text("Replay \(Int(drive.replayRate))×")
                                .font(.subheadline)
                        }
                        .disabled(drive.running)
                    }
                }
            }
            Text("Command bus")
                .font(.caption.weight(.semibold))
            ScrollView(.horizontal, showsIndicators: false) {
                Text(drive.commandLog.isEmpty ? "Frames show up here after Go." : drive.commandLog)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 54)
            Button(action: toggleRun) {
                Text(drive.running ? "Stop" : "Go")
                    .font(.title3.weight(.semibold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .tint(drive.running ? .red : Theme.amber)
            .disabled(!drive.running && drive.route.count < 2)
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .padding(.horizontal, 12)
        .padding(.bottom, 6)
    }

    private func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline.monospacedDigit().weight(.semibold))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var replayBinding: Binding<Bool> {
        Binding(
            get: { drive.replay },
            set: { drive.replay = $0 }
        )
    }

    private func toggleRun() {
        if drive.running {
            drive.stop()
        } else {
            drive.start()
        }
    }

    private func search() async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !searching else { return }
        searching = true
        defer { searching = false }
        do {
            results = try await RouteService.search(query: trimmed, near: origin())
            if results.isEmpty {
                drive.status = "No matches."
            }
        } catch {
            drive.status = "Search failed."
        }
    }

    private func route(to place: PlaceResult) async {
        let start = origin()
        let usingGPS = location.coordinate != nil
        drive.status = "Routing to \(place.name)..."
        do {
            let line = try await RouteService.driveRoute(from: start, to: place.latLon)
            drive.setRoute(line, name: place.name)
            if !usingGPS {
                drive.status = "No GPS yet. Route starts at UC campus. \(Units.meters(drive.routeLengthM)) to \(place.name)."
            }
            results = []
            query = place.name
        } catch {
            drive.status = "No driving route."
        }
    }

    private func origin() -> LatLon {
        location.coordinate ?? DemoOrigin.cincinnati
    }

    private func fit(_ points: [LatLon]) -> MKCoordinateRegion {
        guard let first = points.first else {
            return MKCoordinateRegion(
                center: DemoOrigin.cincinnati.coordinate,
                span: MKCoordinateSpan(latitudeDelta: 0.08, longitudeDelta: 0.08)
            )
        }
        var minLat = first.latitude
        var maxLat = first.latitude
        var minLon = first.longitude
        var maxLon = first.longitude
        for point in points {
            minLat = min(minLat, point.latitude)
            maxLat = max(maxLat, point.latitude)
            minLon = min(minLon, point.longitude)
            maxLon = max(maxLon, point.longitude)
        }
        return MKCoordinateRegion(
            center: CLLocationCoordinate2D(
                latitude: (minLat + maxLat) / 2,
                longitude: (minLon + maxLon) / 2
            ),
            span: MKCoordinateSpan(
                latitudeDelta: max(0.01, (maxLat - minLat) * 1.45),
                longitudeDelta: max(0.01, (maxLon - minLon) * 1.45)
            )
        )
    }
}
