import SwiftUI

struct RootView: View {
    @Environment(\.modelContext) private var modelContext
    @AppStorage("selectedCarID") private var selectedCarID = "honda-accord-2018"
    @StateObject private var drive = DriveController()
    @StateObject private var location = LocationService()

    private var car: CarModel? {
        CarCatalog.car(id: selectedCarID)
    }

    var body: some View {
        TabView {
            DriveMapView()
                .tabItem { Label("Drive", systemImage: "location.fill") }
            GarageView(selectedCarID: $selectedCarID)
                .tabItem { Label("Garage", systemImage: "car.fill") }
            LearnView(selectedCarID: selectedCarID)
                .tabItem { Label("Learn", systemImage: "chart.xyaxis.line") }
        }
        .tint(Theme.amber)
        .preferredColorScheme(.dark)
        .environmentObject(drive)
        .environmentObject(location)
        .onAppear {
            drive.context = modelContext
            drive.locationSource = location
            drive.selectedCar = car
            location.request()
            if let error = CarCatalog.loadError {
                drive.status = error
            }
        }
        .onChange(of: selectedCarID) { _, _ in
            drive.selectedCar = car
        }
    }
}
