#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

using Array3D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray3D = py::array_t<long long, py::array::c_style | py::array::forcecast>;
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

void validate_3d_shape(const py::buffer_info& info, const std::string& name) {
    if (info.ndim != 3) {
        throw std::runtime_error(name + " must be 3D, got " + std::to_string(info.ndim) + "D");
    }
}

void validate_1d_shape(const py::buffer_info& info, const std::string& name, ssize_t expected) {
    if (info.ndim != 1 || info.shape[0] != expected) {
        throw std::runtime_error(
            name + " must have shape (" + std::to_string(expected) + "), got " + shape_to_string(info)
        );
    }
}

auto make_index(ssize_t theta_cells, ssize_t width_zones) {
    return [theta_cells, width_zones](ssize_t r, ssize_t t, ssize_t w) -> ssize_t {
        return ((r * theta_cells) + t) * width_zones + w;
    };
}

std::vector<double> normalized_zone_weights(
    py::handle zone_weights_obj,
    ssize_t width_zone_count
) {
    if (zone_weights_obj.is_none()) {
        return std::vector<double>(static_cast<size_t>(width_zone_count), 1.0 / std::max<ssize_t>(width_zone_count, 1));
    }
    Array1D zone_weights = py::reinterpret_borrow<Array1D>(zone_weights_obj);
    auto zone_info = zone_weights.request();
    validate_1d_shape(zone_info, "zone_weights", width_zone_count);
    const auto* zone_ptr = static_cast<const double*>(zone_info.ptr);
    std::vector<double> output(static_cast<size_t>(width_zone_count), 0.0);
    double total = 0.0;
    for (ssize_t idx = 0; idx < width_zone_count; ++idx) {
        output[static_cast<size_t>(idx)] = zone_ptr[idx];
        total += zone_ptr[idx];
    }
    total = std::max(total, 1e-12);
    for (double& value : output) {
        value /= total;
    }
    return output;
}

std::vector<double> layer_weights(py::handle layer_source_weights_obj) {
    if (layer_source_weights_obj.is_none()) {
        return {};
    }
    Array1D layer_weights = py::reinterpret_borrow<Array1D>(layer_source_weights_obj);
    auto layer_info = layer_weights.request();
    validate_1d_shape(layer_info, "layer_source_weights", 5);
    const auto* layer_ptr = static_cast<const double*>(layer_info.ptr);
    return {
        std::max(layer_ptr[0], 0.0),
        std::max(layer_ptr[1], 0.0),
        std::max(layer_ptr[2], 0.0),
        std::max(layer_ptr[3], 0.0),
        std::max(layer_ptr[4], 0.0),
    };
}

std::vector<double> build_source_field_core(
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones,
    double source_volumetric_fraction,
    double volumetric_source_w_per_m3,
    double wheel_angular_speed_radps,
    double time_s,
    double theta_delta_rad,
    const long long* patch_radial_indices_ptr,
    ssize_t patch_radial_count,
    const long long* theta_offsets_ptr,
    ssize_t theta_offset_count,
    const long long* width_indices_ptr,
    ssize_t width_index_count,
    const long long* layer_index_ptr,
    py::handle zone_weights_obj,
    py::handle layer_source_weights_obj,
    const double* extra_source_ptr
) {
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    auto index = make_index(theta_cells, width_zones);
    std::vector<double> source(static_cast<size_t>(item_count), 0.0);

    const double clamped_source = std::max(volumetric_source_w_per_m3, 0.0);
    const double volumetric_fraction = std::max(source_volumetric_fraction, 0.0);
    const auto active_layer_weights = layer_weights(layer_source_weights_obj);
    if (!active_layer_weights.empty()) {
        double total_weight = 0.0;
        for (double weight : active_layer_weights) {
            total_weight += weight;
        }
        total_weight = std::max(total_weight, 1e-12);
        const double layer_scale = clamped_source * volumetric_fraction;
        for (ssize_t idx = 0; idx < item_count; ++idx) {
            const auto layer_code = static_cast<ssize_t>(layer_index_ptr[idx]);
            if (layer_code == 0 && active_layer_weights[0] > 0.0) {
                source[static_cast<size_t>(idx)] += layer_scale * (active_layer_weights[0] / total_weight);
            } else if (layer_code == 1 && active_layer_weights[1] > 0.0) {
                source[static_cast<size_t>(idx)] += layer_scale * (active_layer_weights[1] / total_weight);
            } else if (layer_code == 2) {
                if (active_layer_weights[2] > 0.0) {
                    source[static_cast<size_t>(idx)] += layer_scale * (active_layer_weights[2] / total_weight);
                }
                if (active_layer_weights[3] > 0.0) {
                    source[static_cast<size_t>(idx)] += layer_scale * (active_layer_weights[3] / total_weight);
                }
            } else if (layer_code == 3 && active_layer_weights[4] > 0.0) {
                source[static_cast<size_t>(idx)] += layer_scale * (active_layer_weights[4] / total_weight);
            }
        }
    } else {
        std::fill(source.begin(), source.end(), clamped_source * volumetric_fraction);
    }

    const double source_remaining = clamped_source * std::max(1.0 - source_volumetric_fraction, 0.0);
    if (source_remaining > 0.0) {
        const double phase_theta = std::fmod(wheel_angular_speed_radps * time_s, kTwoPi);
        const double phase_theta_wrapped = phase_theta >= 0.0 ? phase_theta : phase_theta + kTwoPi;
        const ssize_t center_theta_idx = static_cast<ssize_t>(
            std::nearbyint(phase_theta_wrapped / std::max(theta_delta_rad, 1e-9))
        ) % theta_cells;
        const auto zone_weights = normalized_zone_weights(zone_weights_obj, width_index_count);
        const ssize_t patch_cells = patch_radial_count * theta_offset_count * width_index_count;
        const double patch_extra_density = source_remaining * (radial_cells * theta_cells) / std::max<ssize_t>(patch_cells, 1);
        std::vector<double> patch_delta_by_zone(static_cast<size_t>(width_index_count), 0.0);
        for (ssize_t width_pos = 0; width_pos < width_index_count; ++width_pos) {
            patch_delta_by_zone[static_cast<size_t>(width_pos)] =
                patch_extra_density * zone_weights[static_cast<size_t>(width_pos)];
        }

        for (ssize_t radial_pos = 0; radial_pos < patch_radial_count; ++radial_pos) {
            const ssize_t radial_idx = static_cast<ssize_t>(patch_radial_indices_ptr[radial_pos]);
            if (radial_idx < 0 || radial_idx >= radial_cells) {
                throw std::runtime_error("patch_radial_indices contains an out-of-range index");
            }
            for (ssize_t theta_pos = 0; theta_pos < theta_offset_count; ++theta_pos) {
                const ssize_t theta_idx = (center_theta_idx + static_cast<ssize_t>(theta_offsets_ptr[theta_pos]) + theta_cells) % theta_cells;
                for (ssize_t width_pos = 0; width_pos < width_index_count; ++width_pos) {
                    const ssize_t width_idx = static_cast<ssize_t>(width_indices_ptr[width_pos]);
                    if (width_idx < 0 || width_idx >= width_zones) {
                        throw std::runtime_error("width_indices contains an out-of-range index");
                    }
                    source[static_cast<size_t>(index(radial_idx, theta_idx, width_idx))] +=
                        patch_delta_by_zone[static_cast<size_t>(width_pos)];
                }
            }
        }
    }

    if (extra_source_ptr != nullptr) {
        for (ssize_t idx = 0; idx < item_count; ++idx) {
            source[static_cast<size_t>(idx)] += extra_source_ptr[idx];
        }
    }
    return source;
}

std::pair<std::vector<double>, int> diffuse_vectorized_implicit_core(
    const double* field_ptr,
    const double* source_ptr,
    const double* rho_cp_ptr,
    const double* k_r_ptr,
    const double* k_theta_ptr,
    const double* k_w_ptr,
    double dt_s,
    const double* radial_minus_ptr,
    const double* radial_plus_ptr,
    const double* theta_coeff_ptr,
    const double* width_minus_ptr,
    const double* width_plus_ptr,
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones,
    int diffusion_max_iterations,
    double diffusion_tolerance_k
) {
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    auto index = make_index(theta_cells, width_zones);

    std::vector<double> rhs(static_cast<size_t>(item_count));
    std::vector<double> estimate(static_cast<size_t>(item_count));
    std::vector<double> updated(static_cast<size_t>(item_count));
    std::vector<double> radial_prev(static_cast<size_t>(item_count), 0.0);
    std::vector<double> radial_next(static_cast<size_t>(item_count), 0.0);
    std::vector<double> width_prev(static_cast<size_t>(item_count), 0.0);
    std::vector<double> width_next(static_cast<size_t>(item_count), 0.0);
    std::vector<double> theta_prev(static_cast<size_t>(item_count), 0.0);
    std::vector<double> theta_next(static_cast<size_t>(item_count), 0.0);
    std::vector<double> coeff_r_minus(static_cast<size_t>(item_count));
    std::vector<double> coeff_r_plus(static_cast<size_t>(item_count));
    std::vector<double> coeff_theta(static_cast<size_t>(item_count));
    std::vector<double> coeff_w_minus(static_cast<size_t>(item_count));
    std::vector<double> coeff_w_plus(static_cast<size_t>(item_count));
    std::vector<double> diagonal(static_cast<size_t>(item_count));

    for (ssize_t r = 0; r < radial_cells; ++r) {
        for (ssize_t t = 0; t < theta_cells; ++t) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                const ssize_t idx = index(r, t, w);
                const double rho_safe = std::max(rho_cp_ptr[idx], 1e-12);
                const double alpha_r = k_r_ptr[idx] / rho_safe;
                const double alpha_theta = k_theta_ptr[idx] / rho_safe;
                const double alpha_w = k_w_ptr[idx] / rho_safe;
                rhs[static_cast<size_t>(idx)] = field_ptr[idx] + dt_s * source_ptr[idx] / rho_safe;
                coeff_r_minus[static_cast<size_t>(idx)] = dt_s * alpha_r * radial_minus_ptr[r];
                coeff_r_plus[static_cast<size_t>(idx)] = dt_s * alpha_r * radial_plus_ptr[r];
                coeff_theta[static_cast<size_t>(idx)] = dt_s * alpha_theta * theta_coeff_ptr[r];
                coeff_w_minus[static_cast<size_t>(idx)] = dt_s * alpha_w * width_minus_ptr[w];
                coeff_w_plus[static_cast<size_t>(idx)] = dt_s * alpha_w * width_plus_ptr[w];
                diagonal[static_cast<size_t>(idx)] =
                    1.0
                    + coeff_r_minus[static_cast<size_t>(idx)]
                    + coeff_r_plus[static_cast<size_t>(idx)]
                    + 2.0 * coeff_theta[static_cast<size_t>(idx)]
                    + coeff_w_minus[static_cast<size_t>(idx)]
                    + coeff_w_plus[static_cast<size_t>(idx)];
                estimate[static_cast<size_t>(idx)] = rhs[static_cast<size_t>(idx)];
            }
        }
    }

    int iterations = 0;
    for (iterations = 1; iterations <= std::max(diffusion_max_iterations, 1); ++iterations) {
        std::fill(radial_prev.begin(), radial_prev.end(), 0.0);
        std::fill(radial_next.begin(), radial_next.end(), 0.0);
        std::fill(width_prev.begin(), width_prev.end(), 0.0);
        std::fill(width_next.begin(), width_next.end(), 0.0);

        for (ssize_t r = 1; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 0; w < width_zones; ++w) {
                    radial_prev[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r - 1, t, w))];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells - 1; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 0; w < width_zones; ++w) {
                    radial_next[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r + 1, t, w))];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 1; w < width_zones; ++w) {
                    width_prev[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r, t, w - 1))];
                }
                for (ssize_t w = 0; w < width_zones - 1; ++w) {
                    width_next[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r, t, w + 1))];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                theta_prev[static_cast<size_t>(index(r, 0, w))] = estimate[static_cast<size_t>(index(r, theta_cells - 1, w))];
                for (ssize_t t = 1; t < theta_cells; ++t) {
                    theta_prev[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r, t - 1, w))];
                }
                theta_next[static_cast<size_t>(index(r, theta_cells - 1, w))] = estimate[static_cast<size_t>(index(r, 0, w))];
                for (ssize_t t = 0; t < theta_cells - 1; ++t) {
                    theta_next[static_cast<size_t>(index(r, t, w))] = estimate[static_cast<size_t>(index(r, t + 1, w))];
                }
            }
        }

        double max_delta = 0.0;
        for (ssize_t idx = 0; idx < item_count; ++idx) {
            updated[static_cast<size_t>(idx)] = rhs[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_r_minus[static_cast<size_t>(idx)] * radial_prev[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_r_plus[static_cast<size_t>(idx)] * radial_next[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_theta[static_cast<size_t>(idx)] * theta_prev[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_theta[static_cast<size_t>(idx)] * theta_next[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_w_minus[static_cast<size_t>(idx)] * width_prev[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] += coeff_w_plus[static_cast<size_t>(idx)] * width_next[static_cast<size_t>(idx)];
            updated[static_cast<size_t>(idx)] /= std::max(diagonal[static_cast<size_t>(idx)], 1e-12);
            max_delta = std::max(max_delta, std::abs(updated[static_cast<size_t>(idx)] - estimate[static_cast<size_t>(idx)]));
        }

        estimate.swap(updated);
        if (max_delta < diffusion_tolerance_k) {
            break;
        }
    }
    return {estimate, iterations};
}

py::array_t<double> vector_to_array3d(
    const std::vector<double>& values,
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones
) {
    py::array_t<double> result({radial_cells, theta_cells, width_zones});
    auto result_info = result.request();
    auto* result_ptr = static_cast<double*>(result_info.ptr);
    std::copy(values.begin(), values.end(), result_ptr);
    return result;
}

std::pair<py::array_t<double>, int> diffuse_vectorized_implicit(
    Array3D field,
    Array3D source_w_per_m3,
    Array3D rho_cp,
    Array3D k_r,
    Array3D k_theta,
    Array3D k_w,
    double dt_s,
    Array1D radial_coeff_minus,
    Array1D radial_coeff_plus,
    Array1D theta_coeff,
    Array1D width_coeff_minus,
    Array1D width_coeff_plus,
    int diffusion_max_iterations,
    double diffusion_tolerance_k
) {
    auto field_info = field.request();
    auto source_info = source_w_per_m3.request();
    auto rho_cp_info = rho_cp.request();
    auto k_r_info = k_r.request();
    auto k_theta_info = k_theta.request();
    auto k_w_info = k_w.request();

    validate_3d_shape(field_info, "field");
    validate_3d_shape(source_info, "source_w_per_m3");
    validate_3d_shape(rho_cp_info, "rho_cp");
    validate_3d_shape(k_r_info, "k_r");
    validate_3d_shape(k_theta_info, "k_theta");
    validate_3d_shape(k_w_info, "k_w");

    if (source_info.shape != field_info.shape || rho_cp_info.shape != field_info.shape || k_r_info.shape != field_info.shape ||
        k_theta_info.shape != field_info.shape || k_w_info.shape != field_info.shape) {
        throw std::runtime_error("All 3D inputs must have identical shapes");
    }

    const ssize_t radial_cells = field_info.shape[0];
    const ssize_t theta_cells = field_info.shape[1];
    const ssize_t width_zones = field_info.shape[2];

    auto radial_minus_info = radial_coeff_minus.request();
    auto radial_plus_info = radial_coeff_plus.request();
    auto theta_coeff_info = theta_coeff.request();
    auto width_minus_info = width_coeff_minus.request();
    auto width_plus_info = width_coeff_plus.request();

    validate_1d_shape(radial_minus_info, "radial_coeff_minus", radial_cells);
    validate_1d_shape(radial_plus_info, "radial_coeff_plus", radial_cells);
    validate_1d_shape(theta_coeff_info, "theta_coeff", radial_cells);
    validate_1d_shape(width_minus_info, "width_coeff_minus", width_zones);
    validate_1d_shape(width_plus_info, "width_coeff_plus", width_zones);

    const auto [values, iterations] = diffuse_vectorized_implicit_core(
        static_cast<const double*>(field_info.ptr),
        static_cast<const double*>(source_info.ptr),
        static_cast<const double*>(rho_cp_info.ptr),
        static_cast<const double*>(k_r_info.ptr),
        static_cast<const double*>(k_theta_info.ptr),
        static_cast<const double*>(k_w_info.ptr),
        dt_s,
        static_cast<const double*>(radial_minus_info.ptr),
        static_cast<const double*>(radial_plus_info.ptr),
        static_cast<const double*>(theta_coeff_info.ptr),
        static_cast<const double*>(width_minus_info.ptr),
        static_cast<const double*>(width_plus_info.ptr),
        radial_cells,
        theta_cells,
        width_zones,
        diffusion_max_iterations,
        diffusion_tolerance_k
    );
    return {vector_to_array3d(values, radial_cells, theta_cells, width_zones), iterations};
}

py::array_t<double> build_source_field(
    int radial_cells,
    int theta_cells,
    int width_zones,
    double source_volumetric_fraction,
    double volumetric_source_w_per_m3,
    double wheel_angular_speed_radps,
    double time_s,
    double theta_delta_rad,
    IntArray1D patch_radial_indices,
    IntArray1D theta_offsets,
    IntArray1D width_indices,
    IntArray3D layer_index,
    py::object zone_weights,
    py::object layer_source_weights
) {
    auto patch_radial_info = patch_radial_indices.request();
    auto theta_offsets_info = theta_offsets.request();
    auto width_indices_info = width_indices.request();
    auto layer_index_info = layer_index.request();

    validate_1d_shape(width_indices_info, "width_indices", width_zones);
    validate_3d_shape(layer_index_info, "layer_index");
    if (layer_index_info.shape[0] != radial_cells || layer_index_info.shape[1] != theta_cells || layer_index_info.shape[2] != width_zones) {
        throw std::runtime_error("layer_index must match (radial_cells, theta_cells, width_zones)");
    }

    const auto values = build_source_field_core(
        radial_cells,
        theta_cells,
        width_zones,
        source_volumetric_fraction,
        volumetric_source_w_per_m3,
        wheel_angular_speed_radps,
        time_s,
        theta_delta_rad,
        static_cast<const long long*>(patch_radial_info.ptr),
        patch_radial_info.shape[0],
        static_cast<const long long*>(theta_offsets_info.ptr),
        theta_offsets_info.shape[0],
        static_cast<const long long*>(width_indices_info.ptr),
        width_indices_info.shape[0],
        static_cast<const long long*>(layer_index_info.ptr),
        zone_weights,
        layer_source_weights,
        nullptr
    );
    return vector_to_array3d(values, radial_cells, theta_cells, width_zones);
}

std::tuple<py::array_t<double>, int, py::array_t<double>> build_source_and_diffuse_implicit(
    Array3D field,
    py::object extra_source_w_per_m3,
    Array3D rho_cp,
    Array3D k_r,
    Array3D k_theta,
    Array3D k_w,
    double dt_s,
    Array1D radial_coeff_minus,
    Array1D radial_coeff_plus,
    Array1D theta_coeff,
    Array1D width_coeff_minus,
    Array1D width_coeff_plus,
    int diffusion_max_iterations,
    double diffusion_tolerance_k,
    double source_volumetric_fraction,
    double volumetric_source_w_per_m3,
    double wheel_angular_speed_radps,
    double time_s,
    double theta_delta_rad,
    IntArray1D patch_radial_indices,
    IntArray1D theta_offsets,
    IntArray1D width_indices,
    IntArray3D layer_index,
    py::object zone_weights,
    py::object layer_source_weights
) {
    auto field_info = field.request();
    auto rho_cp_info = rho_cp.request();
    auto k_r_info = k_r.request();
    auto k_theta_info = k_theta.request();
    auto k_w_info = k_w.request();
    auto radial_minus_info = radial_coeff_minus.request();
    auto radial_plus_info = radial_coeff_plus.request();
    auto theta_coeff_info = theta_coeff.request();
    auto width_minus_info = width_coeff_minus.request();
    auto width_plus_info = width_coeff_plus.request();
    auto patch_radial_info = patch_radial_indices.request();
    auto theta_offsets_info = theta_offsets.request();
    auto width_indices_info = width_indices.request();
    auto layer_index_info = layer_index.request();

    validate_3d_shape(field_info, "field");
    validate_3d_shape(rho_cp_info, "rho_cp");
    validate_3d_shape(k_r_info, "k_r");
    validate_3d_shape(k_theta_info, "k_theta");
    validate_3d_shape(k_w_info, "k_w");
    validate_3d_shape(layer_index_info, "layer_index");

    if (rho_cp_info.shape != field_info.shape || k_r_info.shape != field_info.shape || k_theta_info.shape != field_info.shape ||
        k_w_info.shape != field_info.shape || layer_index_info.shape != field_info.shape) {
        throw std::runtime_error("All 3D inputs must have identical shapes");
    }

    const ssize_t radial_cell_count = field_info.shape[0];
    const ssize_t theta_cell_count = field_info.shape[1];
    const ssize_t width_zone_count = field_info.shape[2];

    validate_1d_shape(radial_minus_info, "radial_coeff_minus", radial_cell_count);
    validate_1d_shape(radial_plus_info, "radial_coeff_plus", radial_cell_count);
    validate_1d_shape(theta_coeff_info, "theta_coeff", radial_cell_count);
    validate_1d_shape(width_minus_info, "width_coeff_minus", width_zone_count);
    validate_1d_shape(width_plus_info, "width_coeff_plus", width_zone_count);
    validate_1d_shape(width_indices_info, "width_indices", width_zone_count);

    const double* extra_source_ptr = nullptr;
    py::buffer_info extra_source_info;
    if (!extra_source_w_per_m3.is_none()) {
        Array3D extra_source = py::reinterpret_borrow<Array3D>(extra_source_w_per_m3);
        extra_source_info = extra_source.request();
        validate_3d_shape(extra_source_info, "extra_source_w_per_m3");
        if (extra_source_info.shape != field_info.shape) {
            throw std::runtime_error("extra_source_w_per_m3 must match field shape");
        }
        extra_source_ptr = static_cast<const double*>(extra_source_info.ptr);
    }

    const auto source_values = build_source_field_core(
        radial_cell_count,
        theta_cell_count,
        width_zone_count,
        source_volumetric_fraction,
        volumetric_source_w_per_m3,
        wheel_angular_speed_radps,
        time_s,
        theta_delta_rad,
        static_cast<const long long*>(patch_radial_info.ptr),
        patch_radial_info.shape[0],
        static_cast<const long long*>(theta_offsets_info.ptr),
        theta_offsets_info.shape[0],
        static_cast<const long long*>(width_indices_info.ptr),
        width_indices_info.shape[0],
        static_cast<const long long*>(layer_index_info.ptr),
        zone_weights,
        layer_source_weights,
        extra_source_ptr
    );

    const auto [field_values, iterations] = diffuse_vectorized_implicit_core(
        static_cast<const double*>(field_info.ptr),
        source_values.data(),
        static_cast<const double*>(rho_cp_info.ptr),
        static_cast<const double*>(k_r_info.ptr),
        static_cast<const double*>(k_theta_info.ptr),
        static_cast<const double*>(k_w_info.ptr),
        dt_s,
        static_cast<const double*>(radial_minus_info.ptr),
        static_cast<const double*>(radial_plus_info.ptr),
        static_cast<const double*>(theta_coeff_info.ptr),
        static_cast<const double*>(width_minus_info.ptr),
        static_cast<const double*>(width_plus_info.ptr),
        radial_cell_count,
        theta_cell_count,
        width_zone_count,
        diffusion_max_iterations,
        diffusion_tolerance_k
    );

    return std::make_tuple(
        vector_to_array3d(field_values, radial_cell_count, theta_cell_count, width_zone_count),
        iterations,
        vector_to_array3d(source_values, radial_cell_count, theta_cell_count, width_zone_count)
    );
}

}  // namespace

PYBIND11_MODULE(_hf_diffusion_native, m) {
    m.doc() = "Native high-fidelity diffusion kernel";
    m.def("diffuse_vectorized_implicit", &diffuse_vectorized_implicit);
    m.def("build_source_field", &build_source_field);
    m.def("build_source_and_diffuse_implicit", &build_source_and_diffuse_implicit);
}
