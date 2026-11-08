import Charts
import SwiftData
import SwiftUI

struct SteerChartPoint: Identifiable {
    let id: String
    let curvature: Double
    let steerWheelDeg: Double
    let source: String
}

struct LearnView: View {
    var selectedCarID: String
    @Query(sort: \DriveSession.startedAt, order: .reverse) private var sessions: [DriveSession]
    @Query(sort: \SteerSample.time, order: .forward) private var allSamples: [SteerSample]
    @Environment(\.modelContext) private var context
    @State private var exportURL: URL?

    private var car: CarModel? {
        CarCatalog.car(id: selectedCarID)
    }

    private var carSamples: [SteerSample] {
        allSamples.filter { $0.carId == selectedCarID }
    }

    private var chartPoints: [SteerChartPoint] {
        carSamples.suffix(600).enumerated().map { index, sample in
            SteerChartPoint(
                id: "\(sample.time.timeIntervalSinceReferenceDate)-\(index)",
                curvature: sample.curvature,
                steerWheelDeg: sample.steerWheelDeg,
                source: sample.source
            )
        }
    }

    var body: some View {
        NavigationStack {
            List {
                Section("This car") {
                    if let car {
                        Text(car.displayName)
                            .font(.headline)
                        Text(String(
                            format: "Catalog gain %.0f ° per 1/m. That is wheelbase × steering ratio, small-angle.",
                            BicycleModel.theoreticalGain(car: car)
                        ))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        if let slope = fittedSlope {
                            Text(String(format: "Fitted from this phone: %.0f ° per 1/m", slope))
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Theme.amber)
                        } else {
                            Text("Drive a route with a few turns. The fit shows up after the wheel actually moves.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Text("\(carSamples.count) samples · \(driverCount) from the on-screen wheel")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Section("Steer vs curvature") {
                    if chartPoints.isEmpty {
                        Text("Go on the Drive tab. Each push is saved with the car, the map point, and the wheel angle.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    } else {
                        Chart(chartPoints) { point in
                            PointMark(
                                x: .value("Curvature", point.curvature),
                                y: .value("Wheel", point.steerWheelDeg)
                            )
                            .foregroundStyle(by: .value("Source", point.source))
                        }
                        .chartXAxisLabel("Curvature 1/m")
                        .chartYAxisLabel("Wheel deg")
                        .chartLegend(position: .bottom)
                        .frame(height: 240)
                    }
                }
                Section("Sessions") {
                    if sessions.isEmpty {
                        Text("No drives yet.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(sessions) { session in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(session.carName)
                            Text(session.startedAt, format: .dateTime.month().day().hour().minute())
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text("\(session.samples.count) samples")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                    .onDelete(perform: deleteSessions)
                }
            }
            .navigationTitle("Learn")
            .toolbar {
                if let exportURL {
                    ShareLink(item: exportURL) {
                        Label("Export", systemImage: "square.and.arrow.up")
                    }
                }
            }
            .onAppear { refreshExport() }
            .onChange(of: allSamples.count) { _, _ in refreshExport() }
            .onChange(of: selectedCarID) { _, _ in refreshExport() }
        }
    }

    private var driverCount: Int {
        carSamples.filter { $0.source == "driver" }.count
    }

    private var fittedSlope: Double? {
        OnlineRegressor.fit(
            curvature: carSamples.map(\.curvature),
            steer: carSamples.map(\.steerWheelDeg)
        ).slope
    }

    private func deleteSessions(at offsets: IndexSet) {
        let doomed = offsets.map { sessions[$0] }
        for session in doomed {
            context.delete(session)
        }
        try? context.save()
    }

    private func refreshExport() {
        let rows = carSamples.map { sample in
            ExportRow(
                time: sample.time,
                latitude: sample.latitude,
                longitude: sample.longitude,
                speedMps: sample.speedMps,
                headingDeg: sample.headingDeg,
                curvature: sample.curvature,
                steerWheelDeg: sample.steerWheelDeg,
                throttle: sample.throttle,
                brake: sample.brake,
                lookaheadS: sample.lookaheadS,
                source: sample.source,
                carId: sample.carId
            )
        }
        guard !rows.isEmpty else {
            exportURL = nil
            return
        }
        let payload = TelemetryExport(carId: selectedCarID, rows: rows)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(payload) else {
            exportURL = nil
            return
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("cheesecc-\(selectedCarID).json")
        do {
            try data.write(to: url, options: .atomic)
            exportURL = url
        } catch {
            exportURL = nil
        }
    }
}

private struct TelemetryExport: Codable {
    var carId: String
    var rows: [ExportRow]
}

private struct ExportRow: Codable {
    var time: Date
    var latitude: Double
    var longitude: Double
    var speedMps: Double
    var headingDeg: Double
    var curvature: Double
    var steerWheelDeg: Double
    var throttle: Double
    var brake: Double
    var lookaheadS: Double
    var source: String
    var carId: String
}
