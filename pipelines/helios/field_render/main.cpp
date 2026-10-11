// Render one tile of a field of plots (each a clump of sorghum shoots on soil, 3 ft apart) from a
// table of per-plot inputs, seen from 50 m above in 5 MicaSense RedEdge bands at 63 pixels per
// plot. Writes the tile image as float32 (6 x ny x nx: canopy height in m, then blue, green, red,
// red edge and NIR camera values). keep is not used here: the caller crops the kept plots; the
// others (a ring around the tile) only give neighbors.
//
// Usage: field_render plots.tsv out.bin seed
// Run it in its build directory: Helios loads its assets by relative paths.
// plots.tsv: one header line, then per plot:
//   keep col row shoots leaf_scale chlorophyll senesced dx dy soil_scale
//   col, row: plot position in the tile (0-based); shoots: number of shoots; leaf_scale: leaf
//   prototype scale (sorghum default 0.65); chlorophyll: ug/cm2 of the green leaves; senesced:
//   number of leaves turned senescent; dx, dy: clump offset from the plot center (m); soil_scale:
//   factor on the soil reflectance spectrum.
#include "RadiationModel.h"
#include "PlantArchitecture.h"
#include "LeafOptics.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace helios;

static const float PLOT = 0.9144f; // 3 ft
static const int PX = 63;          // pixels per plot side
static const float CAMERA_HEIGHT = 50.f;
static const float AGE = 20.f;            // days: canopy cover 0.07 to 0.6, neither empty nor closed
static const uint SAMPLES = 5;            // camera rays per pixel
static const uint SCATTERING_DEPTH = 1;
static const size_t DIRECT_RAYS = 10;     // per primitive (Helios default 100)
static const size_t DIFFUSE_RAYS = 50;    // per primitive (Helios default 1000)
static const float DIFFUSE_FLUX = 400.f;  // W/m2 of the diffuse sky: soft shadows
// The mesh simplifications below (leaf and stem subdivisions 4, no petiole geometry) render within
// the Monte Carlo noise of the full sorghum mesh at these ray counts.

struct Plot {
    int keep, col, row, shoots, senesced;
    float leaf_scale, chlorophyll, dx, dy, soil_scale;
};

static double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: field_render plots.tsv out.bin seed\n");
        return 1;
    }
    uint seed = std::atoi(argv[3]);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<Plot> plots;
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);
    int ncol = 0, nrow = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream s(line);
        Plot p;
        s >> p.keep >> p.col >> p.row >> p.shoots >> p.leaf_scale >> p.chlorophyll >> p.senesced >> p.dx >> p.dy >> p.soil_scale;
        plots.push_back(p);
        ncol = std::max(ncol, p.col + 1);
        nrow = std::max(nrow, p.row + 1);
    }
    if (ncol != nrow) helios_runtime_error("field_render: the tile must be square");

    Context context;
    context.seedRandomGenerator(seed);
    context.loadXML("plugins/radiation/spectral_data/soil_surface_spectral_library.xml", true);
    std::vector<vec2> soil_spectrum;
    context.getGlobalData("soil_reflectivity_0000", soil_spectrum);

    PlantArchitecture plantarch(&context);
    plantarch.loadPlantModelFromLibrary("sorghum");
    ShootParameters shoot = plantarch.getCurrentShootParameters("mainstem");
    shoot.phytomer_parameters.leaf.prototype.subdivisions = 4;     // renders as the default 50
    shoot.phytomer_parameters.internode.radial_subdivisions = 4;   // renders as the default 10
    shoot.phytomer_parameters.peduncle.radial_subdivisions = 4;
    shoot.phytomer_parameters.petiole.radial_subdivisions = 0;     // radius 0, length 2 mm: no geometry

    LeafOptics leafoptics(&context);
    LeafOpticsProperties green;
    green.watermass = 0.015f;
    green.drymass = 0.009f;
    LeafOpticsProperties senescent = green;
    senescent.chlorophyllcontent = 3.f;
    senescent.carotenoidcontent = 3.f;
    senescent.anthocyancontent = 0.f;
    senescent.brownpigments = 0.3f;
    std::vector<uint> senescent_leaves;

    for (size_t k = 0; k < plots.size(); k++) {
        const Plot &p = plots[k];
        float cx = (p.col - 0.5f * (ncol - 1)) * PLOT, cy = (p.row - 0.5f * (nrow - 1)) * PLOT;
        // soil: one tile per plot, 50 x 50 patches (shading is per patch; fewer give blocky shadows)
        std::vector<uint> soil = context.addTile(make_vec3(cx, cy, 0), make_vec2(PLOT, PLOT), nullrotation, make_int2(50, 50));
        std::vector<vec2> spectrum = soil_spectrum;
        for (auto &w : spectrum) w.y *= p.soil_scale;
        std::string soil_label = "soil_" + std::to_string(k);
        context.setGlobalData(soil_label.c_str(), spectrum);
        context.setPrimitiveData(soil, "reflectivity_spectrum", soil_label);
        context.setPrimitiveData(soil, "twosided_flag", 0u);
        // the clump: shoots uniform on a disk whose area grows with the number of shoots
        shoot.phytomer_parameters.leaf.prototype_scale = p.leaf_scale;
        plantarch.updateCurrentShootParameters("mainstem", shoot);
        float radius = 0.03f * std::sqrt(float(std::max(p.shoots, 1)));
        std::vector<uint> plant_ids;
        for (int s = 0; s < p.shoots; s++) {
            float r = radius * std::sqrt(context.randu()), a = 2 * M_PI * context.randu();
            plant_ids.push_back(plantarch.buildPlantInstanceFromLibrary(make_vec3(cx + p.dx + r * std::cos(a), cy + p.dy + r * std::sin(a), 0), AGE));
        }
        std::vector<uint> leaf_objects = plantarch.getPlantLeafObjectIDs(plant_ids);
        // senescence: whole leaves drawn at random
        for (int s = 0; s < p.senesced && !leaf_objects.empty(); s++) {
            size_t pick = context.randu(0, int(leaf_objects.size()) - 1);
            std::vector<uint> u = context.getObjectPrimitiveUUIDs(leaf_objects[pick]);
            senescent_leaves.insert(senescent_leaves.end(), u.begin(), u.end());
            leaf_objects.erase(leaf_objects.begin() + pick);
        }
        green.chlorophyllcontent = p.chlorophyll;
        if (!leaf_objects.empty()) leafoptics.run(context.getObjectPrimitiveUUIDs(leaf_objects), green, "leaf_" + std::to_string(k));
    }
    if (!senescent_leaves.empty()) leafoptics.run(senescent_leaves, senescent, "senescent");
    printf("build: %.2f s, %zu plots, %zu primitives\n", secs(t0), plots.size(), context.getPrimitiveCount());

    auto t1 = std::chrono::steady_clock::now();
    RadiationModel radiation(&context);
    radiation.disableMessages();
    uint sun = radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(60), deg2rad(30)));
    radiation.setSourceSpectrum(sun, "solar_spectrum_ASTMG173");
    radiation.setDiffuseSpectrum("solar_spectrum_ASTMG173");
    radiation.setDiffuseSpectrumIntegral(DIFFUSE_FLUX);
    // MicaSense RedEdge band centers and widths
    std::vector<std::string> bands = {"blue", "green", "red", "rededge", "nir"};
    float lo[] = {459, 547, 660, 711, 814}, hi[] = {491, 573, 676, 723, 870};
    for (int b = 0; b < 5; b++) {
        radiation.addRadiationBand(bands[b], lo[b], hi[b]);
        radiation.disableEmission(bands[b]);
        radiation.setScatteringDepth(bands[b], SCATTERING_DEPTH);
        radiation.setDirectRayCount(bands[b], DIRECT_RAYS);
        radiation.setDiffuseRayCount(bands[b], DIFFUSE_RAYS);
    }
    CameraProperties cam;
    cam.camera_resolution = make_int2(PX * ncol, PX * nrow);
    cam.HFOV = 2 * rad2deg(std::atan(0.5f * PLOT * ncol / CAMERA_HEIGHT));
    cam.lens_diameter = 0.f;
    radiation.addRadiationCamera("multispec", bands, make_vec3(0, 0, CAMERA_HEIGHT), make_vec3(0, 0, 0), cam, SAMPLES);
    radiation.updateGeometry();
    radiation.runBand(bands);
    printf("render: %.2f s\n", secs(t1));

    // Canopy height in the band pixel order. The depth ray goes through the pixel center only and
    // can slip through a triangle edge (depth -1): fill such a pixel with the median height of its
    // valid 3 x 3 neighbors.
    int nx = cam.camera_resolution.x, ny = cam.camera_resolution.y, n_miss = 0;
    std::vector<float> ray_depth;
    context.getGlobalData("camera_multispec_pixel_depth", ray_depth);
    std::vector<float> h(ray_depth.size()), height(ray_depth.size());
    for (size_t k = 0; k < ray_depth.size(); k++) h[k] = ray_depth[k] > 0 ? CAMERA_HEIGHT - ray_depth[k] : NAN;
    for (int j = 0; j < ny; j++)
        for (int i = 0; i < nx; i++) {
            height[j * nx + i] = h[j * nx + i];
            if (!std::isnan(h[j * nx + i])) continue;
            n_miss++;
            std::vector<float> nb;
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++) {
                    int jj = j + dj, ii = i + di;
                    if (jj >= 0 && jj < ny && ii >= 0 && ii < nx && !std::isnan(h[jj * nx + ii])) nb.push_back(h[jj * nx + ii]);
                }
            std::sort(nb.begin(), nb.end());
            height[j * nx + i] = nb.empty() ? 0.f : nb[nb.size() / 2];
        }
    std::vector<std::vector<float>> channels = {height};
    for (auto &b : bands) channels.push_back(radiation.getCameraPixelData("multispec", b));

    // The whole tile, 6 x ny x nx, in camera pixel order: pixel (i, j) runs along -x and -y, so
    // plot (col, row) is the block at i0 = (ncol - 1 - col) * 63, j0 = (nrow - 1 - row) * 63
    // (checked with marked plots). The caller crops (after any blur across plot borders).
    std::ofstream out(argv[2], std::ios::binary);
    for (auto &ch : channels) out.write(reinterpret_cast<const char *>(ch.data()), ch.size() * sizeof(float));
    printf("wrote %d x %d tile; height: %d of %d pixels filled; total %.2f s\n", ncol, nrow, n_miss, nx * ny, secs(t0));
    return 0;
}
