import Combine
import CoreLocation
import Foundation

final class LocationService: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published var coordinate: LatLon?
    @Published var speedMps: Double = -1
    @Published var courseDeg: Double = -1
    @Published var compassDeg: Double = -1
    @Published var authorization: CLAuthorizationStatus = .notDetermined
    @Published var statusText = "Waiting for GPS"

    private let manager = CLLocationManager()

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyBest
        manager.distanceFilter = 2
        manager.headingFilter = 5
        manager.activityType = .automotiveNavigation
        manager.pausesLocationUpdatesAutomatically = false
    }

    func request() {
        authorization = manager.authorizationStatus
        manager.requestWhenInUseAuthorization()
        manager.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            manager.startUpdatingHeading()
        }
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        authorization = manager.authorizationStatus
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            statusText = coordinate == nil ? "GPS on" : statusText
            manager.startUpdatingLocation()
        case .denied, .restricted:
            statusText = "Location off. Allow location while using the app."
        case .notDetermined:
            statusText = "Allow location to use live GPS."
        @unknown default:
            statusText = "GPS"
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        coordinate = LatLon(latitude: location.coordinate.latitude, longitude: location.coordinate.longitude)
        if location.speed >= 0 {
            speedMps = location.speed
            statusText = String(format: "GPS %.0f mph", location.speed * 2.2369362921)
        } else {
            speedMps = 0
            statusText = "GPS lock"
        }
        if location.course >= 0 {
            courseDeg = location.course
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        if newHeading.trueHeading >= 0 {
            compassDeg = newHeading.trueHeading
        } else if newHeading.magneticHeading >= 0 {
            compassDeg = newHeading.magneticHeading
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        statusText = error.localizedDescription
    }
}
