import SwiftUI

struct GarageView: View {
    @Binding var selectedCarID: String

    private var selected: CarModel? {
        CarCatalog.car(id: selectedCarID)
    }

    var body: some View {
        NavigationStack {
            List {
                if let error = CarCatalog.loadError, CarCatalog.cars.isEmpty {
                    Section("Catalog") {
                        Text(error)
                    }
                }
                if let car = selected {
                    Section(car.displayName) {
                        Group {
                            if car.id == "hyundai-elantra-2019" {
                                Image("Elantra2019")
                                    .resizable()
                                    .scaledToFit()
                            } else {
                                ChassisDiagramView(car: car, steerWheelDeg: 0)
                            }
                        }
                        .frame(height: 180)
                        .frame(maxWidth: .infinity)
                        .listRowInsets(EdgeInsets(top: 8, leading: 12, bottom: 8, trailing: 12))
                        spec("Body", car.bodyStyle)
                        spec("Size", String(format: "%.2f × %.2f × %.2f m", car.lengthM, car.widthM, car.heightM))
                        spec("Wheelbase", String(format: "%.2f m", car.wheelbaseM))
                        spec("Track", String(format: "%.0f / %.0f mm", car.frontTrackMM, car.rearTrackMM))
                        spec("Mass", String(format: "%.0f kg", car.curbWeightKG))
                        spec("Engine", engineLine(car))
                        spec("Power", String(format: "%.0f hp · %.0f Nm", car.horsepower, car.torqueNm))
                        spec("Power / tonne", String(format: "%.0f hp/t", car.horsepowerPerTonne))
                        spec("Drive", car.drivetrain)
                        spec("Drag", String(format: "Cd %.2f · %.2f m²", car.dragCoefficient, car.frontalAreaM2))
                        spec("Steering", String(format: "%.1f:1 · ±%.0f°", car.steeringRatio, car.maxSteerWheelDeg))
                        spec("Tire radius", String(format: "%.3f m", car.tireRadiusM))
                        spec("Rear-axle turn", String(format: "%.1f m", car.rearAxleTurnRadiusM))
                        spec("Steer gain", String(format: "%.0f ° per 1/m", BicycleModel.theoreticalGain(car: car)))
                    }
                }
                Section("Catalog") {
                    ForEach(CarCatalog.cars) { car in
                        Button {
                            selectedCarID = car.id
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(car.displayName)
                                    Text("\(car.engineType) · \(car.drivetrain) · \(Int(car.horsepower)) hp")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if car.id == selectedCarID {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(Theme.amber)
                                }
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
            .navigationTitle("Garage")
            .safeAreaInset(edge: .bottom) {
                Text("Published-spec approximations. Add a car by editing cars.json.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial)
            }
        }
    }

    private func spec(_ title: String, _ value: String) -> some View {
        LabeledContent(title) {
            Text(value)
                .multilineTextAlignment(.trailing)
        }
    }

    private func engineLine(_ car: CarModel) -> String {
        if let liters = car.displacementL, liters > 0 {
            return String(format: "%@ · %.1f L", car.engineType, liters)
        }
        return car.engineType
    }
}
