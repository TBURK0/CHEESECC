import Foundation

enum CarCatalog {
    static var cars: [CarModel] { loaded.cars }
    static var loadError: String? { loaded.error }

    static func car(id: String) -> CarModel? {
        cars.first { $0.id == id } ?? cars.first
    }

    private static let loaded: (cars: [CarModel], error: String?) = {
        guard let url = CarCatalog.findJSON() else {
            return (cars: [], error: "cars.json was not in the app bundle.")
        }
        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode([CarModel].self, from: data)
            if decoded.isEmpty {
                return (cars: [], error: "cars.json decoded to an empty catalog.")
            }
            return (cars: decoded, error: nil)
        } catch {
            return (cars: [], error: error.localizedDescription)
        }
    }()

    private static func findJSON() -> URL? {
        let bundle = Bundle.main
        if let url = bundle.url(forResource: "cars", withExtension: "json") {
            return url
        }
        if let url = bundle.url(forResource: "cars", withExtension: "json", subdirectory: "Data/Catalog") {
            return url
        }
        guard let resourcePath = bundle.resourcePath else { return nil }
        let root = URL(fileURLWithPath: resourcePath, isDirectory: true)
        guard let enumerator = FileManager.default.enumerator(
            at: root,
            includingPropertiesForKeys: nil
        ) else {
            return nil
        }
        while let url = enumerator.nextObject() as? URL {
            if url.lastPathComponent == "cars.json" {
                return url
            }
        }
        return nil
    }
}
