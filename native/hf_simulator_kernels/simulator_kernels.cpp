#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using Array2D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray1D = py::array_t<long long, py::array::c_style | py::array::forcecast>;

std::string shape_to_string(const py::buffer_info& info) {
    std::string output = "(";
    for (ssize_t idx = 0; idx < info.ndim; ++idx) {
        output += std::to_string(info.shape[idx]);
        if (idx + 1 < info.ndim) {
            output += ", ";
        }
    }
    output += ")";
    return output;
}

void validate_2d_shape(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 2) {
        throw std::runtime_error(name + " must be 2D, got " + std::to_string(info.ndim) + "D");
    }
}

void validate_1d_shape(const py::buffer_info& info, const std::string& name, ssize_t expected) {
    if (info.ndim != 1 || info.shape[0] != expected) {
        throw std::runtime_error(
            name + " must have shape (" + std::to_string(expected) + "), got " + shape_to_string(info)
        );
    }
}

py::array_t<double> step_flash_field(
    Array2D flash_field_tw_k,
    Array2D surface_cell_areas_tw,
    Array1D bulk_surface_w_k,
    Array1D road_surface_temp_w_k,
    Array1D zone_weights,
    IntArray1D theta_indices,
    IntArray1D width_indices,
    double ambient_temp_k,
    double friction_to_tire_w,
    double friction_fraction,
    double bulk_coupling_time_s,
    double ambient_cooling_time_s,
    double patch_relaxation_time_s,
    double road_cooling_time_s,
    double areal_heat_capacity_j_per_m2k,
    double max_delta_above_bulk_k,
    double dt_s
) {
    auto flash_info = flash_field_tw_k.request();
    auto surface_info = surface_cell_areas_tw.request();
    auto bulk_info = bulk_surface_w_k.request();
    auto road_info = road_surface_temp_w_k.request();
    auto zone_weights_info = zone_weights.request();
    auto theta_info = theta_indices.request();
    auto width_info = width_indices.request();

    validate_2d_shape(flash_info, "flash_field_tw_k");
    validate_2d_shape(surface_info, "surface_cell_areas_tw");
    if (surface_info.shape[0] != flash_info.shape[0] || surface_info.shape[1] != flash_info.shape[1]) {
        throw std::runtime_error("surface_cell_areas_tw must match flash_field_tw_k shape");
    }

    const ssize_t theta_cells = flash_info.shape[0];
    const ssize_t width_zones = flash_info.shape[1];
    validate_1d_shape(bulk_info, "bulk_surface_w_k", width_zones);
    validate_1d_shape(road_info, "road_surface_temp_w_k", width_zones);
    validate_1d_shape(zone_weights_info, "zone_weights", width_zones);

    if (theta_info.ndim != 1 || width_info.ndim != 1) {
        throw std::runtime_error("theta_indices and width_indices must be 1D");
    }
    if (width_info.shape[0] != width_zones) {
        throw std::runtime_error("width_indices must match width zone count");
    }

    const auto* flash_ptr = static_cast<const double*>(flash_info.ptr);
    const auto* surface_ptr = static_cast<const double*>(surface_info.ptr);
    const auto* bulk_ptr = static_cast<const double*>(bulk_info.ptr);
    const auto* road_ptr = static_cast<const double*>(road_info.ptr);
    const auto* zone_weights_ptr = static_cast<const double*>(zone_weights_info.ptr);
    const auto* theta_ptr = static_cast<const long long*>(theta_info.ptr);
    const auto* width_ptr = static_cast<const long long*>(width_info.ptr);

    py::array_t<double> result({theta_cells, width_zones});
    auto result_info = result.request();
    auto* result_ptr = static_cast<double*>(result_info.ptr);
    std::copy(flash_ptr, flash_ptr + theta_cells * width_zones, result_ptr);

    std::vector<unsigned char> theta_in_patch(static_cast<size_t>(theta_cells), 0);
    for (ssize_t idx = 0; idx < theta_info.shape[0]; ++idx) {
        const auto theta_idx = static_cast<ssize_t>(theta_ptr[idx]);
        if (theta_idx < 0 || theta_idx >= theta_cells) {
            throw std::runtime_error("theta_indices contains an out-of-range index");
        }
        theta_in_patch[static_cast<size_t>(theta_idx)] = 1;
    }

    auto index = [width_zones](ssize_t theta_idx, ssize_t width_idx) -> ssize_t {
        return theta_idx * width_zones + width_idx;
    };

    const double bulk_coupling_inv = 1.0 / std::max(bulk_coupling_time_s, 1e-6);
    const double ambient_cooling_inv = 1.0 / std::max(ambient_cooling_time_s, 1e-6);
    const double patch_relax_inv = 1.0 / std::max(patch_relaxation_time_s, 1e-6);
    const double road_cooling_inv = 1.0 / std::max(road_cooling_time_s, 1e-6);
    const double heat_capacity_safe = std::max(areal_heat_capacity_j_per_m2k, 1e-9);

    for (ssize_t zone_idx = 0; zone_idx < width_zones; ++zone_idx) {
        const auto width_idx = static_cast<ssize_t>(width_ptr[zone_idx]);
        if (width_idx < 0 || width_idx >= width_zones) {
            throw std::runtime_error("width_indices contains an out-of-range index");
        }

        double zone_patch_area_m2 = 0.0;
        for (ssize_t theta_slot = 0; theta_slot < theta_info.shape[0]; ++theta_slot) {
            zone_patch_area_m2 += surface_ptr[index(static_cast<ssize_t>(theta_ptr[theta_slot]), width_idx)];
        }
        zone_patch_area_m2 = std::max(zone_patch_area_m2, 1e-9);
        const double q_flash_zone = friction_fraction * friction_to_tire_w * zone_weights_ptr[zone_idx];
        const double q_patch_flux_w_per_m2 = q_flash_zone / zone_patch_area_m2;
        double min_flash_excess_k = 0.0;
        if (q_flash_zone > 0.0) {
            min_flash_excess_k = 1.5 + 20.0 * std::min(q_patch_flux_w_per_m2 / 120000.0, 1.5);
        }
        const double bulk_temp_k = bulk_ptr[width_idx];
        const double floor_temp_k = std::min(bulk_temp_k, ambient_temp_k) - 15.0;

        for (ssize_t theta_idx = 0; theta_idx < theta_cells; ++theta_idx) {
            const ssize_t cell_idx = index(theta_idx, width_idx);
            const double area_m2 = std::max(surface_ptr[cell_idx], 1e-9);
            const double temp_k = result_ptr[cell_idx];
            const double q_bulk = area_m2 * (bulk_temp_k - temp_k) * bulk_coupling_inv;
            const double q_ambient = area_m2 * (ambient_temp_k - temp_k) * ambient_cooling_inv;
            double q_patch = 0.0;
            double q_road = 0.0;
            double q_relax = area_m2 * (bulk_temp_k - temp_k) * patch_relax_inv;
            if (theta_in_patch[static_cast<size_t>(theta_idx)] != 0) {
                q_patch = q_flash_zone * area_m2 / zone_patch_area_m2;
                q_road = area_m2 * (road_ptr[width_idx] - temp_k) * road_cooling_inv;
                q_relax *= 0.35;
            }
            const double delta_k = dt_s * (q_patch + q_bulk + q_ambient + q_road + q_relax) /
                                   std::max(heat_capacity_safe * area_m2, 1e-9);
            double temp_next_k = temp_k + delta_k;
            if (theta_in_patch[static_cast<size_t>(theta_idx)] != 0 && q_flash_zone > 0.0) {
                temp_next_k = std::max(temp_next_k, bulk_temp_k + min_flash_excess_k);
            }
            temp_next_k = std::min(temp_next_k, bulk_temp_k + max_delta_above_bulk_k);
            temp_next_k = std::max(temp_next_k, floor_temp_k);
            result_ptr[cell_idx] = temp_next_k;
        }
    }

    return result;
}

std::pair<py::array_t<double>, double> step_sidewall_field(
    Array2D sidewall_field_tw_k,
    Array1D shoulder_temp_by_zone_k,
    double gas_temp_k,
    double ambient_temp_k,
    double rim_temp_k,
    double solar_w_m2,
    double wind_mps,
    double wind_yaw_rad,
    double wheel_wake_factor,
    double brake_heat_to_sidewall_w,
    double dt_s
) {
    auto sidewall_info = sidewall_field_tw_k.request();
    auto shoulder_info = shoulder_temp_by_zone_k.request();
    validate_2d_shape(sidewall_info, "sidewall_field_tw_k");
    const ssize_t theta_cells = sidewall_info.shape[0];
    const ssize_t width_zones = sidewall_info.shape[1];
    validate_1d_shape(shoulder_info, "shoulder_temp_by_zone_k", width_zones);

    const auto* sidewall_ptr = static_cast<const double*>(sidewall_info.ptr);
    const auto* shoulder_ptr = static_cast<const double*>(shoulder_info.ptr);

    py::array_t<double> result({theta_cells, width_zones});
    auto result_info = result.request();
    auto* result_ptr = static_cast<double*>(result_info.ptr);
    std::copy(sidewall_ptr, sidewall_ptr + theta_cells * width_zones, result_ptr);

    auto index = [width_zones](ssize_t theta_idx, ssize_t zone_idx) -> ssize_t {
        return theta_idx * width_zones + zone_idx;
    };

    const double per_zone_brake = brake_heat_to_sidewall_w / std::max<ssize_t>(width_zones, 1);
    const double q_brake = per_zone_brake / std::max<ssize_t>(theta_cells, 1);
    const double capacity = 1.65e5;
    const double h_amb_base = 18.0 + 4.5 * wind_mps * (1.0 + 0.12 * std::abs(std::sin(wind_yaw_rad)));
    const double h_amb = h_amb_base * (1.0 + 0.15 * std::max(wheel_wake_factor - 1.0, -0.8));

    double total_heat_w = 0.0;
    for (ssize_t zone_idx = 0; zone_idx < width_zones; ++zone_idx) {
        const double shoulder_bias = zone_idx != 1 ? 1.10 : 0.85;
        const double shoulder_temp_k = shoulder_ptr[zone_idx];
        const double q_solar = zone_idx != 1 ? 0.06 * solar_w_m2 : 0.04 * solar_w_m2;
        for (ssize_t theta_idx = 0; theta_idx < theta_cells; ++theta_idx) {
            const ssize_t cell_idx = index(theta_idx, zone_idx);
            const double temp = result_ptr[cell_idx];
            const double q_carcass = 72.0 * shoulder_bias * (shoulder_temp_k - temp);
            const double q_rim = 24.0 * (rim_temp_k - temp);
            const double q_gas = 16.0 * (gas_temp_k - temp);
            const double q_amb = h_amb * (temp - ambient_temp_k);
            const double delta = dt_s * (q_carcass + q_rim + q_gas + q_solar + q_brake - q_amb) /
                                 std::max(capacity, 1.0);
            result_ptr[cell_idx] = temp + delta;
            total_heat_w += q_carcass + q_rim + q_gas + q_brake;
        }
    }

    return {result, total_heat_w / std::max<ssize_t>(theta_cells, 1)};
}

}  // namespace

PYBIND11_MODULE(_hf_simulator_native, m) {
    m.doc() = "Native high-fidelity simulator kernels";
    m.def("step_flash_field", &step_flash_field);
    m.def("step_sidewall_field", &step_sidewall_field);
}
