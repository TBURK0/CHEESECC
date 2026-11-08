import CoreLocation
import Foundation
import MapKit

struct PlaceResult: Identifiable {
    let id = UUID()
    var name: String
    var latitude: Double
    var longitude: Double

    var latLon: LatLon { LatLon(latitude: latitude, longitude: longitude) }
}

enum RouteError: LocalizedError {
    case noRoute

    var errorDescription: String? {
        "MapKit did not return a road route."
    }
}

enum RouteService {
    static func search(query: String, near: LatLon) async throws -> [PlaceResult] {
        let request = MKLocalSearch.Request()
        request.naturalLanguageQuery = query
        request.region = MKCoordinateRegion(
            center: near.coordinate,
            span: MKCoordinateSpan(latitudeDelta: 0.6, longitudeDelta: 0.6)
        )
        let response = try await MKLocalSearch(request: request).start()
        return response.mapItems.prefix(6).map { item in
            PlaceResult(
                name: item.name ?? "Place",
                latitude: item.placemark.coordinate.latitude,
                longitude: item.placemark.coordinate.longitude
            )
        }
    }

    static func driveRoute(from: LatLon, to: LatLon) async throws -> [LatLon] {
        let request = MKDirections.Request()
        request.source = MKMapItem(placemark: MKPlacemark(coordinate: from.coordinate))
        request.destination = MKMapItem(placemark: MKPlacemark(coordinate: to.coordinate))
        request.transportType = .automobile
        let response = try await MKDirections(request: request).calculate()
        guard let route = response.routes.first, route.polyline.pointCount >= 2 else {
            throw RouteError.noRoute
        }
        return route.polyline.vertexCoordinates.map {
            LatLon(latitude: $0.latitude, longitude: $0.longitude)
        }
    }
}

extension LatLon {
    var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: latitude, longitude: longitude)
    }
}

extension MKPolyline {
    var vertexCoordinates: [CLLocationCoordinate2D] {
        var coordinates = [CLLocationCoordinate2D](
            repeating: CLLocationCoordinate2D(latitude: 0, longitude: 0),
            count: pointCount
        )
        getCoordinates(&coordinates, range: NSRange(location: 0, length: pointCount))
        return coordinates
    }
}
