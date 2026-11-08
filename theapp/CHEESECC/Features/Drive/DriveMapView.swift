import MapKit
import SwiftUI

struct DriveMapView: View {
    @EnvironmentObject private var drive: DriveController
    @EnvironmentObject private var location: LocationService
    @State private var fromText = ""
    @State private var toText = ""
    @State private var fromPlace: PlaceResult?
    @State private var toPlace: PlaceResult?
    @State private var activeField: Field = .to
    @State private var results: [PlaceResult] = []
    @State private var searching = false
    @State private var searchTask: Task<Void, Never>?
    @State private var searchToken = 0
    @State private var routeToken = 0
    @State private var routeInFlight = false
    @State private var routeRetryAfter = Date.distantPast
    @State private var handwheel = 0.0
    @State private var holdingWheel = false
    @FocusState private var focused: Field?
    @State private var camera: MapCameraPosition = .region(
        MKCoordinateRegion(
            center: DemoOrigin.cincinnati.coordinate,
            span: MKCoordinateSpan(latitudeDelta: 0.08, longitudeDelta: 0.08)
        )
    )

    private enum Field: Hashable {
        case from
        case to
    }

    private var shownCar: CarModel? {
        drive.activeCar ?? drive.selectedCar
    }

    var body: some View {
        Map(position: $camera, interactionModes: drive.running ? MapInteractionModes() : .all) {
            UserAnnotation()
            if drive.route.count > 1 {
                MapPolyline(coordinates: drive.route.map(\.coordinate))
                    .stroke(Theme.amber, lineWidth: 5)
                if let start = drive.route.first {
                    Marker("Start", coordinate: start.coordinate)
                        .tint(.green)
                }
            }
            if let destination = drive.destination, drive.route.count > 1 {
                Marker(
                    drive.destinationName.isEmpty ? "Destination" : drive.destinationName,
                    coordinate: destination.coordinate
                )
                .tint(.red)
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
        .onChange(of: drive.route) { _, route in
            guard !drive.running else { return }
            if route.count > 1 {
                camera = .region(fit(route))
            } else if let coordinate = location.coordinate {
                camera = .region(region(around: coordinate))
            }
        }
        .onChange(of: drive.running) { _, running in
            if running {
                followCamera()
            } else if drive.route.count > 1 {
                camera = .region(fit(drive.route))
            }
        }
        .onChange(of: drive.position) { _, _ in
            followCamera()
        }
        .onChange(of: drive.headingDeg) { _, _ in
            followCamera()
        }
        .onChange(of: location.coordinate?.latitude) { _, _ in
            guard let coordinate = location.coordinate else { return }
            if fromPlace == nil, toPlace != nil, drive.route.count < 2, !drive.running {
                Task { await routeIfReady(force: false) }
                return
            }
            guard drive.route.isEmpty, !drive.running else { return }
            camera = .region(region(around: coordinate))
        }
        .onChange(of: drive.plannerSteerDeg) { _, newValue in
            if !holdingWheel {
                handwheel = newValue
            }
        }
        .onChange(of: focused) { _, newValue in
            guard let newValue, newValue != activeField else { return }
            activeField = newValue
            results = []
            searchTask?.cancel()
        }
    }

    private var searchCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Real-Time Route Planning & Dynamic Lookahead")
                .font(.subheadline.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Text(shownCar?.displayName ?? "No car")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(location.statusText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            VStack(spacing: 0) {
                placeField(
                    title: "From",
                    text: $fromText,
                    placeholder: "Current Location",
                    field: .from,
                    showClear: fromPlace != nil || !fromText.isEmpty,
                    onClear: { resetFrom() }
                )
                Divider().padding(.leading, 28)
                placeField(
                    title: "To",
                    text: $toText,
                    placeholder: "Destination",
                    field: .to,
                    showClear: toPlace != nil || !toText.isEmpty,
                    onClear: { resetTo() }
                )
            }
            .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
            .onChange(of: fromText) { _, newValue in
                fieldTextChanged(.from, text: newValue)
            }
            .onChange(of: toText) { _, newValue in
                fieldTextChanged(.to, text: newValue)
            }
            if searching && results.isEmpty {
                Text("Searching…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(Array(results.prefix(6))) { place in
                Button(place.name) { choose(place) }
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

    private func placeField(
        title: String,
        text: Binding<String>,
        placeholder: String,
        field: Field,
        showClear: Bool,
        onClear: @escaping () -> Void
    ) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(field == .from ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(focused == field ? Theme.amber : .secondary)
                .frame(width: 36, alignment: .leading)
            TextField(placeholder, text: text)
                .textFieldStyle(.roundedBorder)
                .textInputAutocapitalization(.words)
                .submitLabel(.search)
                .focused($focused, equals: field)
                .onSubmit { submitSearch() }
            if showClear {
                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(field == .from ? "Current location" : "Clear destination")
            } else {
                Color.clear.frame(width: 24, height: 24)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func toggleRun() {
        if drive.running {
            drive.stop()
        } else {
            drive.start()
        }
    }

    private func fieldTextChanged(_ field: Field, text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            searchTask?.cancel()
            results = []
            switch field {
            case .from:
                fromPlace = nil
                Task { await routeIfReady(force: true) }
            case .to:
                let hadDestination = toPlace != nil || drive.destination != nil || drive.route.count > 1
                toPlace = nil
                if hadDestination {
                    drive.clearRoute()
                }
            }
            return
        }
        switch field {
        case .from:
            if text == fromPlace?.name { return }
            guard focused == .from else { return }
            fromPlace = nil
        case .to:
            if text == toPlace?.name { return }
            guard focused == .to else { return }
            toPlace = nil
        }
        activeField = field
        scheduleSearch()
    }

    private func resetFrom() {
        searchTask?.cancel()
        searchToken += 1
        fromPlace = nil
        results = []
        if fromText.isEmpty {
            Task { await routeIfReady(force: true) }
        } else {
            fromText = ""
        }
    }

    private func resetTo() {
        searchTask?.cancel()
        searchToken += 1
        if toText.isEmpty {
            toPlace = nil
            if drive.route.count > 1 || drive.destination != nil {
                drive.clearRoute()
            }
        } else {
            toText = ""
        }
    }

    private func scheduleSearch() {
        searchTask?.cancel()
        let field = focused ?? activeField
        let raw = field == .from ? fromText : toText
        let query = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            results = []
            return
        }
        searchTask = Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            guard !Task.isCancelled else { return }
            await runSearch(query: query, field: field)
        }
    }

    private func submitSearch() {
        searchTask?.cancel()
        let field = focused ?? activeField
        activeField = field
        let query = (field == .from ? fromText : toText).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        Task { await runSearch(query: query, field: field) }
    }

    private func runSearch(query: String, field: Field) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        searchToken += 1
        let token = searchToken
        searching = true
        defer {
            if token == searchToken {
                searching = false
            }
        }
        do {
            let found = try await RouteService.search(query: trimmed, near: searchBias())
            guard token == searchToken, activeField == field else { return }
            let current = (field == .from ? fromText : toText).trimmingCharacters(in: .whitespacesAndNewlines)
            guard current == trimmed else { return }
            results = Array(found.prefix(6))
            if results.isEmpty {
                drive.status = "No matches."
            }
        } catch {
            guard token == searchToken else { return }
            drive.status = "Search failed."
        }
    }

    private func choose(_ place: PlaceResult) {
        searchTask?.cancel()
        searchToken += 1
        switch activeField {
        case .from:
            fromPlace = place
            fromText = place.name
        case .to:
            toPlace = place
            toText = place.name
        }
        results = []
        Task { await routeIfReady(force: true) }
    }

    private func routeIfReady(force: Bool) async {
        guard let destination = toPlace else { return }
        if !force, routeInFlight || Date() < routeRetryAfter {
            return
        }

        let origin: LatLon
        if drive.running {
            guard let fix = location.coordinate else {
                drive.status = "Location is still coming."
                return
            }
            origin = fix
        } else if let fromPlace {
            origin = fromPlace.latLon
        } else if let fix = location.coordinate {
            origin = fix
        } else {
            if drive.route.count > 1 || drive.destination != nil {
                drive.clearRoute()
            }
            drive.status = "Location is still coming."
            return
        }

        routeToken += 1
        let token = routeToken
        routeInFlight = true
        defer {
            if routeToken == token {
                routeInFlight = false
            }
        }

        let name = destination.name
        let target = destination.latLon
        drive.status = "Routing to \(name)..."
        do {
            let line = try await RouteService.driveRoute(from: origin, to: target)
            guard routeToken == token else { return }
            if drive.running {
                drive.replaceRoute(line, name: name, destination: target)
            } else {
                drive.setRoute(line, name: name, destination: target)
            }
            results = []
        } catch {
            guard routeToken == token else { return }
            drive.status = "No driving route."
            if !force {
                routeRetryAfter = Date().addingTimeInterval(5)
            }
        }
    }

    /// Bias for place search only. Driving routes never start at the demo campus pin.
    private func searchBias() -> LatLon {
        if let fromPlace { return fromPlace.latLon }
        if let fix = location.coordinate { return fix }
        if let toPlace { return toPlace.latLon }
        return DemoOrigin.cincinnati
    }

    private func region(around coordinate: LatLon) -> MKCoordinateRegion {
        MKCoordinateRegion(
            center: coordinate.coordinate,
            span: MKCoordinateSpan(latitudeDelta: 0.04, longitudeDelta: 0.04)
        )
    }

    private func followCamera() {
        guard drive.running else { return }
        let next = MapCamera(
            centerCoordinate: drive.position.coordinate,
            distance: 520,
            heading: drive.headingDeg,
            pitch: 40
        )
        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            camera = .camera(next)
        }
    }

    private func fit(_ points: [LatLon]) -> MKCoordinateRegion {
        var fitted = points
        if let destination = drive.destination {
            fitted.append(destination)
        }
        guard let first = fitted.first else {
            return MKCoordinateRegion(
                center: DemoOrigin.cincinnati.coordinate,
                span: MKCoordinateSpan(latitudeDelta: 0.08, longitudeDelta: 0.08)
            )
        }
        var minLat = first.latitude
        var maxLat = first.latitude
        var minLon = first.longitude
        var maxLon = first.longitude
        for point in fitted {
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
