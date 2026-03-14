#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
using IntArray2D = py::array_t<long long, py::array::c_style | py::array::forcecast>;

enum class DiffusionSolverMode {
    Legacy = 0,
    Adi = 1,
    Auto = 2,
};

struct LineWorkspace {
    std::vector<double> lower;
    std::vector<double> diagonal;
    std::vector<double> upper;
    std::vector<double> rhs;
    std::vector<double> helper_rhs;
    std::vector<double> helper_solution;
    std::vector<double> modified_diagonal;

    explicit LineWorkspace(ssize_t max_line_size = 1) {
        ensure_size(max_line_size);
    }

    void ensure_size(ssize_t max_line_size) {
        const size_t line_size = static_cast<size_t>(std::max<ssize_t>(max_line_size, 1));
        const size_t off_diag_size = static_cast<size_t>(std::max<ssize_t>(max_line_size - 1, 0));
        lower.resize(off_diag_size);
        diagonal.resize(line_size);
        upper.resize(off_diag_size);
        rhs.resize(line_size);
        helper_rhs.resize(line_size);
        helper_solution.resize(line_size);
        modified_diagonal.resize(line_size);
    }
};

struct DiffusionWorkspace {
    std::vector<double> current;
    std::vector<double> advected;
    std::vector<double> rhs;
    std::vector<double> estimate;
    std::vector<double> radial_sweep;
    std::vector<double> theta_sweep;
    std::vector<double> updated;
    std::vector<double> coeff_r_minus;
    std::vector<double> coeff_r_plus;
    std::vector<double> coeff_theta;
    std::vector<double> coeff_w_minus;
    std::vector<double> coeff_w_plus;
    std::vector<double> diagonal;
    std::vector<double> radial_diagonal;
    std::vector<double> theta_diagonal;
    std::vector<double> width_diagonal;
    std::vector<double> neg_coeff_r_minus;
    std::vector<double> neg_coeff_r_plus;
    std::vector<double> neg_coeff_theta;
    std::vector<double> neg_coeff_w_minus;
    std::vector<double> neg_coeff_w_plus;
    LineWorkspace line;

    void ensure_size(ssize_t item_count, ssize_t max_line_size) {
        const size_t size = static_cast<size_t>(std::max<ssize_t>(item_count, 1));
        current.resize(size);
        advected.resize(size);
        rhs.resize(size);
        estimate.resize(size);
        radial_sweep.resize(size);
        theta_sweep.resize(size);
        updated.resize(size);
        coeff_r_minus.resize(size);
        coeff_r_plus.resize(size);
        coeff_theta.resize(size);
        coeff_w_minus.resize(size);
        coeff_w_plus.resize(size);
        diagonal.resize(size);
        radial_diagonal.resize(size);
        theta_diagonal.resize(size);
        width_diagonal.resize(size);
        neg_coeff_r_minus.resize(size);
        neg_coeff_r_plus.resize(size);
        neg_coeff_theta.resize(size);
        neg_coeff_w_minus.resize(size);
        neg_coeff_w_plus.resize(size);
        line.ensure_size(max_line_size);
    }
};

struct SourceWorkspace {
    std::vector<double> source;
    std::vector<double> zone_weights;
    std::vector<double> patch_delta_by_zone;
    std::vector<double> layer_weights;

    void ensure_size(ssize_t item_count, ssize_t width_zone_count) {
        source.resize(static_cast<size_t>(std::max<ssize_t>(item_count, 1)));
        zone_weights.resize(static_cast<size_t>(std::max<ssize_t>(width_zone_count, 1)));
        patch_delta_by_zone.resize(static_cast<size_t>(std::max<ssize_t>(width_zone_count, 1)));
        layer_weights.resize(5);
    }
};

struct CompactPropertyWorkspace {
    std::vector<double> rho_cp_r;
    std::vector<double> k_r_rw;
    std::vector<double> k_theta_rw;
    std::vector<double> k_w_rw;
    std::vector<double> coeff_r_minus_rw;
    std::vector<double> coeff_r_plus_rw;
    std::vector<double> coeff_theta_rw;
    std::vector<double> coeff_w_minus_rw;
    std::vector<double> coeff_w_plus_rw;
    std::vector<double> radial_diagonal_rw;
    std::vector<double> theta_diagonal_rw;
    std::vector<double> width_diagonal_rw;
    std::vector<long long> radial_layer_codes;

    void ensure_size(ssize_t radial_cells, ssize_t width_zones) {
        rho_cp_r.resize(static_cast<size_t>(std::max<ssize_t>(radial_cells, 1)));
        const size_t radial_width_size = static_cast<size_t>(std::max<ssize_t>(radial_cells * width_zones, 1));
        k_r_rw.resize(radial_width_size);
        k_theta_rw.resize(radial_width_size);
        k_w_rw.resize(radial_width_size);
        coeff_r_minus_rw.resize(radial_width_size);
        coeff_r_plus_rw.resize(radial_width_size);
        coeff_theta_rw.resize(radial_width_size);
        coeff_w_minus_rw.resize(radial_width_size);
        coeff_w_plus_rw.resize(radial_width_size);
        radial_diagonal_rw.resize(radial_width_size);
        theta_diagonal_rw.resize(radial_width_size);
        width_diagonal_rw.resize(radial_width_size);
        radial_layer_codes.resize(static_cast<size_t>(std::max<ssize_t>(radial_cells, 1)));
    }
};

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

void validate_2d_shape(
    const py::buffer_info& info,
    const std::string& name,
    ssize_t expected_rows,
    ssize_t expected_cols
) {
    if (info.ndim != 2 || info.shape[0] != expected_rows || info.shape[1] != expected_cols) {
        throw std::runtime_error(
            name + " must have shape ("
            + std::to_string(expected_rows)
            + ", "
            + std::to_string(expected_cols)
            + "), got "
            + shape_to_string(info)
        );
    }
}

auto make_index(ssize_t theta_cells, ssize_t width_zones) {
    return [theta_cells, width_zones](ssize_t r, ssize_t t, ssize_t w) -> ssize_t {
        return ((r * theta_cells) + t) * width_zones + w;
    };
}

void normalized_zone_weights(
    py::handle zone_weights_obj,
    ssize_t width_zone_count,
    std::vector<double>& output
) {
    output.resize(static_cast<size_t>(std::max<ssize_t>(width_zone_count, 1)));
    if (zone_weights_obj.is_none()) {
        std::fill(
            output.begin(),
            output.begin() + static_cast<size_t>(width_zone_count),
            1.0 / std::max<ssize_t>(width_zone_count, 1)
        );
        return;
    }
    Array1D zone_weights = py::reinterpret_borrow<Array1D>(zone_weights_obj);
    auto zone_info = zone_weights.request();
    validate_1d_shape(zone_info, "zone_weights", width_zone_count);
    const auto* zone_ptr = static_cast<const double*>(zone_info.ptr);
    double total = 0.0;
    for (ssize_t idx = 0; idx < width_zone_count; ++idx) {
        output[static_cast<size_t>(idx)] = zone_ptr[idx];
        total += zone_ptr[idx];
    }
    total = std::max(total, 1e-12);
    for (double& value : output) {
        value /= total;
    }
}

bool layer_weights(py::handle layer_source_weights_obj, std::vector<double>& output) {
    output.resize(5);
    if (layer_source_weights_obj.is_none()) {
        return false;
    }
    Array1D layer_weights = py::reinterpret_borrow<Array1D>(layer_source_weights_obj);
    auto layer_info = layer_weights.request();
    validate_1d_shape(layer_info, "layer_source_weights", 5);
    const auto* layer_ptr = static_cast<const double*>(layer_info.ptr);
    for (ssize_t idx = 0; idx < 5; ++idx) {
        output[static_cast<size_t>(idx)] = std::max(layer_ptr[idx], 0.0);
    }
    return true;
}

SourceWorkspace& source_workspace() {
    thread_local SourceWorkspace workspace;
    return workspace;
}

DiffusionWorkspace& diffusion_workspace() {
    thread_local DiffusionWorkspace workspace;
    return workspace;
}

CompactPropertyWorkspace& compact_property_workspace() {
    thread_local CompactPropertyWorkspace workspace;
    return workspace;
}

DiffusionSolverMode parse_solver_mode(const std::string& solver_mode) {
    if (solver_mode == "adi") {
        return DiffusionSolverMode::Adi;
    }
    if (solver_mode == "auto") {
        return DiffusionSolverMode::Auto;
    }
    return DiffusionSolverMode::Legacy;
}

void advect_theta_periodic_core(
    const double* input_ptr,
    std::vector<double>& output,
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones,
    double omega,
    double dt_s,
    double theta_delta_rad
) {
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    output.resize(static_cast<size_t>(std::max<ssize_t>(item_count, 1)));
    if (std::abs(omega) <= 1e-12) {
        std::copy_n(input_ptr, item_count, output.begin());
        return;
    }

    auto index = make_index(theta_cells, width_zones);
    const double shift_cells = omega * dt_s / std::max(theta_delta_rad, 1e-12);
    if (shift_cells >= 0.0) {
        const ssize_t base_shift = static_cast<ssize_t>(std::floor(shift_cells));
        const double frac = shift_cells - static_cast<double>(base_shift);
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                const ssize_t primary_t = (t - base_shift % theta_cells + theta_cells) % theta_cells;
                const ssize_t secondary_t = (t - ((base_shift + 1) % theta_cells) + theta_cells) % theta_cells;
                for (ssize_t w = 0; w < width_zones; ++w) {
                    output[static_cast<size_t>(index(r, t, w))] =
                        (1.0 - frac) * input_ptr[index(r, primary_t, w)]
                        + frac * input_ptr[index(r, secondary_t, w)];
                }
            }
        }
        return;
    }

    const double source_shift = -shift_cells;
    const ssize_t base_shift = static_cast<ssize_t>(std::floor(source_shift));
    const double frac = source_shift - static_cast<double>(base_shift);
    for (ssize_t r = 0; r < radial_cells; ++r) {
        for (ssize_t t = 0; t < theta_cells; ++t) {
            const ssize_t primary_t = (t + base_shift) % theta_cells;
            const ssize_t secondary_t = (t + base_shift + 1) % theta_cells;
            for (ssize_t w = 0; w < width_zones; ++w) {
                output[static_cast<size_t>(index(r, t, w))] =
                    (1.0 - frac) * input_ptr[index(r, primary_t, w)]
                    + frac * input_ptr[index(r, secondary_t, w)];
            }
        }
    }
}

void build_radial_layer_codes_from_slices(
    const long long* layer_slices_ptr,
    ssize_t radial_cells,
    std::vector<long long>& radial_layer_codes
) {
    radial_layer_codes.resize(static_cast<size_t>(std::max<ssize_t>(radial_cells, 1)));
    std::fill(radial_layer_codes.begin(), radial_layer_codes.end(), 0);
    for (ssize_t layer_code = 0; layer_code < 4; ++layer_code) {
        const ssize_t start = static_cast<ssize_t>(layer_slices_ptr[layer_code * 2]);
        const ssize_t stop = static_cast<ssize_t>(layer_slices_ptr[layer_code * 2 + 1]);
        for (ssize_t r = std::max<ssize_t>(start, 0); r < std::min<ssize_t>(stop, radial_cells); ++r) {
            radial_layer_codes[static_cast<size_t>(r)] = layer_code;
        }
    }
}

void build_compact_properties(
    CompactPropertyWorkspace& workspace,
    ssize_t radial_cells,
    ssize_t width_zones,
    const long long* layer_slices_ptr,
    const double* volumetric_heat_capacity_by_layer_ptr,
    const double* k_r_base_by_layer_ptr,
    const double* k_theta_base_by_layer_ptr,
    const double* k_w_base_by_layer_ptr,
    const double* shoulder_bias_by_layer_ptr,
    const double* center_bias_by_layer_ptr,
    const double* bead_bias_by_layer_ptr,
    const double* temp_sensitivity_by_layer_ptr,
    const double* wear_sensitivity_by_layer_ptr,
    const double* reinforcement_density_by_layer_ptr,
    const double* cord_angle_deg_by_layer_ptr,
    const double* grain_index_w_ptr,
    const double* blister_index_w_ptr,
    double wear,
    double age_index,
    bool construction_enabled,
    double construction_bead_width_fraction,
    double construction_temp_reference_k,
    double tread_blister_conductivity_penalty
) {
    workspace.ensure_size(radial_cells, width_zones);
    build_radial_layer_codes_from_slices(layer_slices_ptr, radial_cells, workspace.radial_layer_codes);

    const double age_gain = 1.0 + 0.06 * std::max(age_index, 0.0);
    const double wear_clamped = std::clamp(wear, 0.0, 1.0);
    const double bead_limit = std::max(construction_bead_width_fraction, 1e-6);

    std::vector<double> k_r_by_layer_width(static_cast<size_t>(4 * std::max<ssize_t>(width_zones, 1)), 0.0);
    std::vector<double> k_theta_by_layer_width(static_cast<size_t>(4 * std::max<ssize_t>(width_zones, 1)), 0.0);
    std::vector<double> k_w_by_layer_width(static_cast<size_t>(4 * std::max<ssize_t>(width_zones, 1)), 0.0);

    for (ssize_t layer_code = 0; layer_code < 4; ++layer_code) {
        for (ssize_t w = 0; w < width_zones; ++w) {
            double radial_scale = 1.0;
            double theta_scale = 1.0;
            double width_scale = 1.0;
            if (construction_enabled) {
                const double position = width_zones <= 1
                    ? 0.0
                    : -1.0 + 2.0 * static_cast<double>(w) / std::max<ssize_t>(width_zones - 1, 1);
                const double abs_position = std::abs(position);
                double shoulder_weight = std::clamp(abs_position, 0.0, 1.0);
                double center_weight = std::max(1.0 - abs_position, 0.0);
                double bead_weight = std::max(1.0 - std::min(abs_position / bead_limit, 1.0), 0.0);
                const double total_weight = std::max(shoulder_weight + center_weight + bead_weight, 1e-9);
                shoulder_weight /= total_weight;
                center_weight /= total_weight;
                bead_weight /= total_weight;
                const double width_bias =
                    shoulder_weight * shoulder_bias_by_layer_ptr[layer_code]
                    + center_weight * center_bias_by_layer_ptr[layer_code]
                    + bead_weight * bead_bias_by_layer_ptr[layer_code];
                const double temp_gain =
                    1.0 + temp_sensitivity_by_layer_ptr[layer_code] * (construction_temp_reference_k - construction_temp_reference_k);
                const double wear_gain = 1.0 - wear_sensitivity_by_layer_ptr[layer_code] * wear_clamped;
                const double reinforcement_delta = std::max(reinforcement_density_by_layer_ptr[layer_code] - 1.0, 0.0);
                const double angle_rad = cord_angle_deg_by_layer_ptr[layer_code] * (kTwoPi / 360.0);
                radial_scale = 1.0 + 0.12 * reinforcement_delta * std::abs(std::cos(angle_rad));
                theta_scale = 1.0 + 0.18 * reinforcement_delta * std::abs(std::sin(angle_rad));
                width_scale = 1.0 + 0.08 * reinforcement_delta * std::abs(std::sin(2.0 * angle_rad));
                const double common = std::max(width_bias * temp_gain * wear_gain, 0.25);
                radial_scale = std::max(common * radial_scale, 0.25);
                theta_scale = std::max(common * theta_scale, 0.25);
                width_scale = std::max(common * width_scale, 0.25);
            }

            const double grain_penalty = layer_code == 0 ? 1.0 - 0.08 * grain_index_w_ptr[w] : 1.0;
            const double active_blister_penalty = layer_code == 0
                ? 1.0 - tread_blister_conductivity_penalty * blister_index_w_ptr[w]
                : 1.0 - 0.10 * blister_index_w_ptr[w];
            const size_t lw_idx = static_cast<size_t>(layer_code * width_zones + w);
            k_r_by_layer_width[lw_idx] = std::max(
                k_r_base_by_layer_ptr[layer_code] * grain_penalty * active_blister_penalty * radial_scale,
                1e-4
            );
            k_theta_by_layer_width[lw_idx] = std::max(
                k_theta_base_by_layer_ptr[layer_code] * grain_penalty * active_blister_penalty * theta_scale,
                1e-4
            );
            k_w_by_layer_width[lw_idx] = std::max(
                k_w_base_by_layer_ptr[layer_code] * active_blister_penalty * width_scale,
                1e-4
            );
        }
    }

    for (ssize_t r = 0; r < radial_cells; ++r) {
        const auto layer_code = static_cast<ssize_t>(workspace.radial_layer_codes[static_cast<size_t>(r)]);
        workspace.rho_cp_r[static_cast<size_t>(r)] =
            volumetric_heat_capacity_by_layer_ptr[layer_code] * age_gain;
        for (ssize_t w = 0; w < width_zones; ++w) {
            const size_t rw_idx = static_cast<size_t>(r * width_zones + w);
            const size_t lw_idx = static_cast<size_t>(layer_code * width_zones + w);
            workspace.k_r_rw[rw_idx] = k_r_by_layer_width[lw_idx];
            workspace.k_theta_rw[rw_idx] = k_theta_by_layer_width[lw_idx];
            workspace.k_w_rw[rw_idx] = k_w_by_layer_width[lw_idx];
        }
    }
}

const std::vector<double>& build_source_field_core(
    SourceWorkspace& workspace,
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
    workspace.ensure_size(item_count, width_zones);
    auto& source = workspace.source;
    std::fill(source.begin(), source.begin() + static_cast<size_t>(item_count), 0.0);

    const double clamped_source = std::max(volumetric_source_w_per_m3, 0.0);
    const double volumetric_fraction = std::max(source_volumetric_fraction, 0.0);
    const bool has_layer_weights = layer_weights(layer_source_weights_obj, workspace.layer_weights);
    const auto& active_layer_weights = workspace.layer_weights;
    if (has_layer_weights) {
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
        normalized_zone_weights(zone_weights_obj, width_index_count, workspace.zone_weights);
        const auto& zone_weights = workspace.zone_weights;
        const ssize_t patch_cells = patch_radial_count * theta_offset_count * width_index_count;
        const double patch_extra_density = source_remaining * (radial_cells * theta_cells) / std::max<ssize_t>(patch_cells, 1);
        auto& patch_delta_by_zone = workspace.patch_delta_by_zone;
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

const std::vector<double>& build_source_field_core_compressed(
    SourceWorkspace& workspace,
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
    const long long* radial_layer_codes_ptr,
    py::handle zone_weights_obj,
    py::handle layer_source_weights_obj,
    const double* extra_source_ptr
) {
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    auto index = make_index(theta_cells, width_zones);
    workspace.ensure_size(item_count, width_zones);
    auto& source = workspace.source;
    std::fill(source.begin(), source.begin() + static_cast<size_t>(item_count), 0.0);

    const double clamped_source = std::max(volumetric_source_w_per_m3, 0.0);
    const double volumetric_fraction = std::max(source_volumetric_fraction, 0.0);
    const bool has_layer_weights = layer_weights(layer_source_weights_obj, workspace.layer_weights);
    const auto& active_layer_weights = workspace.layer_weights;
    if (has_layer_weights) {
        double total_weight = 0.0;
        for (double weight : active_layer_weights) {
            total_weight += weight;
        }
        total_weight = std::max(total_weight, 1e-12);
        const double layer_scale = clamped_source * volumetric_fraction;
        for (ssize_t r = 0; r < radial_cells; ++r) {
            const auto layer_code = static_cast<ssize_t>(radial_layer_codes_ptr[r]);
            double layer_fraction = 0.0;
            if (layer_code == 0) {
                layer_fraction = active_layer_weights[0] / total_weight;
            } else if (layer_code == 1) {
                layer_fraction = active_layer_weights[1] / total_weight;
            } else if (layer_code == 2) {
                layer_fraction = (active_layer_weights[2] + active_layer_weights[3]) / total_weight;
            } else if (layer_code == 3) {
                layer_fraction = active_layer_weights[4] / total_weight;
            }
            if (layer_fraction <= 0.0) {
                continue;
            }
            const double layer_density = layer_scale * layer_fraction;
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 0; w < width_zones; ++w) {
                    source[static_cast<size_t>(index(r, t, w))] += layer_density;
                }
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
        normalized_zone_weights(zone_weights_obj, width_index_count, workspace.zone_weights);
        const auto& zone_weights = workspace.zone_weights;
        const ssize_t patch_cells = patch_radial_count * theta_offset_count * width_index_count;
        const double patch_extra_density = source_remaining * (radial_cells * theta_cells) / std::max<ssize_t>(patch_cells, 1);
        auto& patch_delta_by_zone = workspace.patch_delta_by_zone;
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

double source_power_w_native(
    const std::vector<double>& source_values,
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones,
    py::handle cell_volumes_m3
) {
    Array3D cell_volumes = py::reinterpret_borrow<Array3D>(cell_volumes_m3);
    auto cell_volumes_info = cell_volumes.request();
    validate_3d_shape(cell_volumes_info, "cell_volumes_m3");
    if (
        cell_volumes_info.shape[0] != radial_cells
        || cell_volumes_info.shape[1] != theta_cells
        || cell_volumes_info.shape[2] != width_zones
    ) {
        throw std::runtime_error("cell_volumes_m3 must match source field shape");
    }
    const auto* cell_volume_ptr = static_cast<const double*>(cell_volumes_info.ptr);
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    double total_power = 0.0;
    for (ssize_t idx = 0; idx < item_count; ++idx) {
        total_power += source_values[static_cast<size_t>(idx)] * cell_volume_ptr[idx];
    }
    return total_power;
}

double field_energy_j_native_compact(
    const double* field_ptr,
    const std::vector<double>& rho_cp_r,
    ssize_t radial_cells,
    ssize_t theta_cells,
    ssize_t width_zones,
    const double* cell_volume_ptr
) {
    auto index = make_index(theta_cells, width_zones);
    double total_energy = 0.0;
    for (ssize_t r = 0; r < radial_cells; ++r) {
        const double rho_cp = rho_cp_r[static_cast<size_t>(r)];
        for (ssize_t t = 0; t < theta_cells; ++t) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                const ssize_t idx = index(r, t, w);
                total_energy += field_ptr[idx] * rho_cp * cell_volume_ptr[idx];
            }
        }
    }
    return total_energy;
}

void solve_tridiagonal_in_place(LineWorkspace& workspace, ssize_t line_size) {
    if (line_size == 1) {
        workspace.rhs[0] /= std::max(workspace.diagonal[0], 1e-12);
        return;
    }

    for (ssize_t idx = 1; idx < line_size; ++idx) {
        const size_t prev = static_cast<size_t>(idx - 1);
        const size_t current = static_cast<size_t>(idx);
        const double pivot = std::max(workspace.diagonal[prev], 1e-12);
        const double factor = workspace.lower[prev] / pivot;
        workspace.diagonal[current] -= factor * workspace.upper[prev];
        workspace.rhs[current] -= factor * workspace.rhs[prev];
    }

    workspace.rhs[static_cast<size_t>(line_size - 1)] /=
        std::max(workspace.diagonal[static_cast<size_t>(line_size - 1)], 1e-12);
    for (ssize_t idx = line_size - 2; idx >= 0; --idx) {
        const size_t current = static_cast<size_t>(idx);
        workspace.rhs[current] =
            (workspace.rhs[current] - workspace.upper[current] * workspace.rhs[current + 1])
            / std::max(workspace.diagonal[current], 1e-12);
    }
}

void solve_tridiagonal_size3_in_place(
    double lower0,
    double lower1,
    double diagonal0,
    double diagonal1,
    double diagonal2,
    double upper0,
    double upper1,
    double& rhs0,
    double& rhs1,
    double& rhs2
) {
    const double pivot0 = std::max(diagonal0, 1e-12);
    const double factor0 = lower0 / pivot0;
    diagonal1 -= factor0 * upper0;
    rhs1 -= factor0 * rhs0;

    const double pivot1 = std::max(diagonal1, 1e-12);
    const double factor1 = lower1 / pivot1;
    diagonal2 -= factor1 * upper1;
    rhs2 -= factor1 * rhs1;

    rhs2 /= std::max(diagonal2, 1e-12);
    rhs1 = (rhs1 - upper1 * rhs2) / std::max(diagonal1, 1e-12);
    rhs0 = (rhs0 - upper0 * rhs1) / std::max(diagonal0, 1e-12);
}

void solve_cyclic_tridiagonal_in_place(
    LineWorkspace& workspace,
    ssize_t line_size,
    double alpha,
    double beta
) {
    if (line_size == 1) {
        workspace.rhs[0] /= std::max(workspace.diagonal[0] + alpha + beta, 1e-12);
        return;
    }

    const double gamma = workspace.diagonal[0] != 0.0 ? -workspace.diagonal[0] : -1.0;
    std::copy_n(workspace.diagonal.begin(), line_size, workspace.modified_diagonal.begin());
    workspace.modified_diagonal[0] -= gamma;
    workspace.modified_diagonal[static_cast<size_t>(line_size - 1)] -= (alpha * beta) / gamma;

    std::copy_n(workspace.modified_diagonal.begin(), line_size, workspace.diagonal.begin());
    solve_tridiagonal_in_place(workspace, line_size);
    std::copy_n(workspace.rhs.begin(), line_size, workspace.helper_solution.begin());

    std::fill_n(workspace.helper_rhs.begin(), line_size, 0.0);
    workspace.helper_rhs[0] = gamma;
    workspace.helper_rhs[static_cast<size_t>(line_size - 1)] = alpha;
    std::copy_n(workspace.helper_rhs.begin(), line_size, workspace.rhs.begin());
    std::copy_n(workspace.modified_diagonal.begin(), line_size, workspace.diagonal.begin());
    solve_tridiagonal_in_place(workspace, line_size);

    const double denominator = std::max(
        1.0 + workspace.rhs[0] + beta * workspace.rhs[static_cast<size_t>(line_size - 1)] / gamma,
        1e-12
    );
    const double factor =
        (workspace.helper_solution[0] + beta * workspace.helper_solution[static_cast<size_t>(line_size - 1)] / gamma)
        / denominator;
    for (ssize_t idx = 0; idx < line_size; ++idx) {
        workspace.helper_solution[static_cast<size_t>(idx)] -= factor * workspace.rhs[static_cast<size_t>(idx)];
    }
    std::copy_n(workspace.helper_solution.begin(), line_size, workspace.rhs.begin());
}

std::pair<const std::vector<double>*, int> diffuse_vectorized_implicit_core(
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
    auto& workspace = diffusion_workspace();
    workspace.ensure_size(item_count, std::max({radial_cells, theta_cells, width_zones}));
    auto& rhs = workspace.rhs;
    auto* estimate = &workspace.estimate;

    for (ssize_t r = 0; r < radial_cells; ++r) {
        for (ssize_t t = 0; t < theta_cells; ++t) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                const ssize_t idx = index(r, t, w);
                const double rho_safe = std::max(rho_cp_ptr[idx], 1e-12);
                rhs[static_cast<size_t>(idx)] = field_ptr[idx] + dt_s * source_ptr[idx] / rho_safe;
                (*estimate)[static_cast<size_t>(idx)] = rhs[static_cast<size_t>(idx)];
            }
        }
    }

    int iterations = 0;
    for (iterations = 1; iterations <= std::max(diffusion_max_iterations, 1); ++iterations) {
        double max_delta = 0.0;
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                const ssize_t t_minus = (t - 1 + theta_cells) % theta_cells;
                const ssize_t t_plus = (t + 1) % theta_cells;
                for (ssize_t w = 0; w < width_zones; ++w) {
                    const ssize_t idx = index(r, t, w);
                    const double rho_safe = std::max(rho_cp_ptr[idx], 1e-12);
                    const double alpha_r = k_r_ptr[idx] / rho_safe;
                    const double alpha_theta = k_theta_ptr[idx] / rho_safe;
                    const double alpha_w = k_w_ptr[idx] / rho_safe;

                    const double coeff_r_minus = alpha_r * radial_minus_ptr[r];
                    const double coeff_r_plus = alpha_r * radial_plus_ptr[r];
                    const double coeff_theta = alpha_theta * theta_coeff_ptr[r];
                    const double coeff_w_minus = alpha_w * width_minus_ptr[w];
                    const double coeff_w_plus = alpha_w * width_plus_ptr[w];

                    double neighbor_sum = 0.0;
                    double diagonal = 1.0;
                    if (r > 0) {
                        neighbor_sum += coeff_r_minus * (*estimate)[static_cast<size_t>(index(r - 1, t, w))];
                        diagonal += dt_s * coeff_r_minus;
                    }
                    if (r + 1 < radial_cells) {
                        neighbor_sum += coeff_r_plus * (*estimate)[static_cast<size_t>(index(r + 1, t, w))];
                        diagonal += dt_s * coeff_r_plus;
                    }

                    neighbor_sum += coeff_theta * (*estimate)[static_cast<size_t>(index(r, t_minus, w))];
                    neighbor_sum += coeff_theta * (*estimate)[static_cast<size_t>(index(r, t_plus, w))];
                    diagonal += 2.0 * dt_s * coeff_theta;

                    if (w > 0) {
                        neighbor_sum += coeff_w_minus * (*estimate)[static_cast<size_t>(index(r, t, w - 1))];
                        diagonal += dt_s * coeff_w_minus;
                    }
                    if (w + 1 < width_zones) {
                        neighbor_sum += coeff_w_plus * (*estimate)[static_cast<size_t>(index(r, t, w + 1))];
                        diagonal += dt_s * coeff_w_plus;
                    }

                    const double prior = (*estimate)[static_cast<size_t>(idx)];
                    const double next =
                        (rhs[static_cast<size_t>(idx)] + dt_s * neighbor_sum) / std::max(diagonal, 1e-12);
                    (*estimate)[static_cast<size_t>(idx)] = next;
                    max_delta = std::max(max_delta, std::abs(next - prior));
                }
            }
        }

        if (max_delta < diffusion_tolerance_k) {
            break;
        }
    }
    return {estimate, iterations};
}

std::pair<const std::vector<double>*, int> diffuse_vectorized_implicit_core_compact(
    const double* field_ptr,
    const double* source_ptr,
    const std::vector<double>& rho_cp_r,
    const std::vector<double>& k_r_rw,
    const std::vector<double>& k_theta_rw,
    const std::vector<double>& k_w_rw,
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
    double diffusion_tolerance_k,
    DiffusionSolverMode solver_mode = DiffusionSolverMode::Legacy
) {
    const ssize_t item_count = radial_cells * theta_cells * width_zones;
    auto index = make_index(theta_cells, width_zones);
    auto& workspace = diffusion_workspace();
    workspace.ensure_size(item_count, std::max({radial_cells, theta_cells, width_zones}));
    auto& compact_workspace = compact_property_workspace();
    auto& rhs = workspace.rhs;
    auto* estimate = &workspace.estimate;

    for (ssize_t r = 0; r < radial_cells; ++r) {
        const double rho_safe = std::max(rho_cp_r[static_cast<size_t>(r)], 1e-12);
        for (ssize_t w = 0; w < width_zones; ++w) {
            const size_t rw_idx = static_cast<size_t>(r * width_zones + w);
            const double alpha_r = k_r_rw[rw_idx] / rho_safe;
            const double alpha_theta = k_theta_rw[rw_idx] / rho_safe;
            const double alpha_w = k_w_rw[rw_idx] / rho_safe;
            compact_workspace.coeff_r_minus_rw[rw_idx] = alpha_r * radial_minus_ptr[r];
            compact_workspace.coeff_r_plus_rw[rw_idx] = alpha_r * radial_plus_ptr[r];
            compact_workspace.coeff_theta_rw[rw_idx] = alpha_theta * theta_coeff_ptr[r];
            compact_workspace.coeff_w_minus_rw[rw_idx] = alpha_w * width_minus_ptr[w];
            compact_workspace.coeff_w_plus_rw[rw_idx] = alpha_w * width_plus_ptr[w];
        }
    }

    for (ssize_t r = 0; r < radial_cells; ++r) {
        const double rho_safe = std::max(rho_cp_r[static_cast<size_t>(r)], 1e-12);
        for (ssize_t t = 0; t < theta_cells; ++t) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                const ssize_t idx = index(r, t, w);
                rhs[static_cast<size_t>(idx)] = field_ptr[idx] + dt_s * source_ptr[idx] / rho_safe;
                (*estimate)[static_cast<size_t>(idx)] = rhs[static_cast<size_t>(idx)];
            }
        }
    }

    const int max_iterations =
        solver_mode == DiffusionSolverMode::Adi
        ? 1
        : std::max(diffusion_max_iterations, 1);
    int iterations = 0;
    for (iterations = 1; iterations <= max_iterations; ++iterations) {
        double max_delta = 0.0;
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                const ssize_t t_minus = (t - 1 + theta_cells) % theta_cells;
                const ssize_t t_plus = (t + 1) % theta_cells;
                for (ssize_t w = 0; w < width_zones; ++w) {
                    const ssize_t idx = index(r, t, w);
                    const size_t rw_idx = static_cast<size_t>(r * width_zones + w);

                    const double coeff_r_minus = compact_workspace.coeff_r_minus_rw[rw_idx];
                    const double coeff_r_plus = compact_workspace.coeff_r_plus_rw[rw_idx];
                    const double coeff_theta = compact_workspace.coeff_theta_rw[rw_idx];
                    const double coeff_w_minus = compact_workspace.coeff_w_minus_rw[rw_idx];
                    const double coeff_w_plus = compact_workspace.coeff_w_plus_rw[rw_idx];

                    double neighbor_sum = 0.0;
                    double diagonal = 1.0;
                    if (r > 0) {
                        neighbor_sum += coeff_r_minus * (*estimate)[static_cast<size_t>(index(r - 1, t, w))];
                        diagonal += dt_s * coeff_r_minus;
                    }
                    if (r + 1 < radial_cells) {
                        neighbor_sum += coeff_r_plus * (*estimate)[static_cast<size_t>(index(r + 1, t, w))];
                        diagonal += dt_s * coeff_r_plus;
                    }

                    neighbor_sum += coeff_theta * (*estimate)[static_cast<size_t>(index(r, t_minus, w))];
                    neighbor_sum += coeff_theta * (*estimate)[static_cast<size_t>(index(r, t_plus, w))];
                    diagonal += 2.0 * dt_s * coeff_theta;

                    if (w > 0) {
                        neighbor_sum += coeff_w_minus * (*estimate)[static_cast<size_t>(index(r, t, w - 1))];
                        diagonal += dt_s * coeff_w_minus;
                    }
                    if (w + 1 < width_zones) {
                        neighbor_sum += coeff_w_plus * (*estimate)[static_cast<size_t>(index(r, t, w + 1))];
                        diagonal += dt_s * coeff_w_plus;
                    }

                    const double prior = (*estimate)[static_cast<size_t>(idx)];
                    const double next =
                        (rhs[static_cast<size_t>(idx)] + dt_s * neighbor_sum) / std::max(diagonal, 1e-12);
                    (*estimate)[static_cast<size_t>(idx)] = next;
                    max_delta = std::max(max_delta, std::abs(next - prior));
                }
            }
        }

        if (max_delta < diffusion_tolerance_k || solver_mode == DiffusionSolverMode::Adi) {
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

    std::pair<const std::vector<double>*, int> diffuse_result;
    {
        py::gil_scoped_release release;
        diffuse_result = diffuse_vectorized_implicit_core(
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
    }
    const auto& values = *diffuse_result.first;
    const int iterations = diffuse_result.second;
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

    auto& source_build_workspace = source_workspace();
    const auto& values = build_source_field_core(
        source_build_workspace,
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

    auto& source_build_workspace = source_workspace();
    const auto& source_values = build_source_field_core(
        source_build_workspace,
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

    std::pair<const std::vector<double>*, int> diffuse_result;
    {
        py::gil_scoped_release release;
        diffuse_result = diffuse_vectorized_implicit_core(
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
    }
    const auto& field_values = *diffuse_result.first;
    const int iterations = diffuse_result.second;

    return std::make_tuple(
        vector_to_array3d(field_values, radial_cell_count, theta_cell_count, width_zone_count),
        iterations,
        vector_to_array3d(source_values, radial_cell_count, theta_cell_count, width_zone_count)
    );
}

std::tuple<py::array_t<double>, int, double, double, double, double, double> thermal_step_multi_substep(
    Array3D field,
    py::object extra_source_w_per_m3,
    Array3D cell_volumes_m3,
    double dt_s,
    int substeps,
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
    IntArray2D layer_slices,
    Array1D volumetric_heat_capacity_by_layer,
    Array1D k_r_base_by_layer,
    Array1D k_theta_base_by_layer,
    Array1D k_w_base_by_layer,
    Array1D shoulder_bias_by_layer,
    Array1D center_bias_by_layer,
    Array1D bead_bias_by_layer,
    Array1D temp_sensitivity_by_layer,
    Array1D wear_sensitivity_by_layer,
    Array1D reinforcement_density_by_layer,
    Array1D cord_angle_deg_by_layer,
    Array1D grain_index_w,
    Array1D blister_index_w,
    double wear,
    double age_index,
    bool construction_enabled,
    double construction_bead_width_fraction,
    double construction_temp_reference_k,
    double tread_blister_conductivity_penalty,
    py::object zone_weights,
    py::object layer_source_weights,
    double minimum_temperature_k,
    double maximum_temperature_k,
    bool enable_profiling,
    const std::string& solver_mode
) {
    auto field_info = field.request();
    auto cell_volumes_info = cell_volumes_m3.request();
    auto radial_minus_info = radial_coeff_minus.request();
    auto radial_plus_info = radial_coeff_plus.request();
    auto theta_coeff_info = theta_coeff.request();
    auto width_minus_info = width_coeff_minus.request();
    auto width_plus_info = width_coeff_plus.request();
    auto patch_radial_info = patch_radial_indices.request();
    auto theta_offsets_info = theta_offsets.request();
    auto width_indices_info = width_indices.request();
    auto layer_slices_info = layer_slices.request();
    auto volumetric_heat_capacity_info = volumetric_heat_capacity_by_layer.request();
    auto k_r_base_info = k_r_base_by_layer.request();
    auto k_theta_base_info = k_theta_base_by_layer.request();
    auto k_w_base_info = k_w_base_by_layer.request();
    auto shoulder_bias_info = shoulder_bias_by_layer.request();
    auto center_bias_info = center_bias_by_layer.request();
    auto bead_bias_info = bead_bias_by_layer.request();
    auto temp_sensitivity_info = temp_sensitivity_by_layer.request();
    auto wear_sensitivity_info = wear_sensitivity_by_layer.request();
    auto reinforcement_density_info = reinforcement_density_by_layer.request();
    auto cord_angle_deg_info = cord_angle_deg_by_layer.request();
    auto grain_index_info = grain_index_w.request();
    auto blister_index_info = blister_index_w.request();

    validate_3d_shape(field_info, "field");
    validate_3d_shape(cell_volumes_info, "cell_volumes_m3");
    if (cell_volumes_info.shape != field_info.shape) {
        throw std::runtime_error("cell_volumes_m3 must match field shape");
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
    validate_2d_shape(layer_slices_info, "layer_slices", 4, 2);
    validate_1d_shape(volumetric_heat_capacity_info, "volumetric_heat_capacity_by_layer", 4);
    validate_1d_shape(k_r_base_info, "k_r_base_by_layer", 4);
    validate_1d_shape(k_theta_base_info, "k_theta_base_by_layer", 4);
    validate_1d_shape(k_w_base_info, "k_w_base_by_layer", 4);
    validate_1d_shape(shoulder_bias_info, "shoulder_bias_by_layer", 4);
    validate_1d_shape(center_bias_info, "center_bias_by_layer", 4);
    validate_1d_shape(bead_bias_info, "bead_bias_by_layer", 4);
    validate_1d_shape(temp_sensitivity_info, "temp_sensitivity_by_layer", 4);
    validate_1d_shape(wear_sensitivity_info, "wear_sensitivity_by_layer", 4);
    validate_1d_shape(reinforcement_density_info, "reinforcement_density_by_layer", 4);
    validate_1d_shape(cord_angle_deg_info, "cord_angle_deg_by_layer", 4);
    validate_1d_shape(grain_index_info, "grain_index_w", width_zone_count);
    validate_1d_shape(blister_index_info, "blister_index_w", width_zone_count);

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

    const double dt_sub = dt_s / std::max(substeps, 1);
    const DiffusionSolverMode parsed_solver_mode = parse_solver_mode(solver_mode);
    const ssize_t item_count = radial_cell_count * theta_cell_count * width_zone_count;
    auto& source_build_workspace = source_workspace();
    auto& workspace = diffusion_workspace();
    auto& compact_workspace = compact_property_workspace();
    build_compact_properties(
        compact_workspace,
        radial_cell_count,
        width_zone_count,
        static_cast<const long long*>(layer_slices_info.ptr),
        static_cast<const double*>(volumetric_heat_capacity_info.ptr),
        static_cast<const double*>(k_r_base_info.ptr),
        static_cast<const double*>(k_theta_base_info.ptr),
        static_cast<const double*>(k_w_base_info.ptr),
        static_cast<const double*>(shoulder_bias_info.ptr),
        static_cast<const double*>(center_bias_info.ptr),
        static_cast<const double*>(bead_bias_info.ptr),
        static_cast<const double*>(temp_sensitivity_info.ptr),
        static_cast<const double*>(wear_sensitivity_info.ptr),
        static_cast<const double*>(reinforcement_density_info.ptr),
        static_cast<const double*>(cord_angle_deg_info.ptr),
        static_cast<const double*>(grain_index_info.ptr),
        static_cast<const double*>(blister_index_info.ptr),
        wear,
        age_index,
        construction_enabled,
        construction_bead_width_fraction,
        construction_temp_reference_k,
        tread_blister_conductivity_penalty
    );
    workspace.ensure_size(item_count, std::max({radial_cell_count, theta_cell_count, width_zone_count}));
    std::copy_n(static_cast<const double*>(field_info.ptr), item_count, workspace.current.begin());

    double total_source_energy_j = 0.0;
    const double initial_energy_j = field_energy_j_native_compact(
        static_cast<const double*>(field_info.ptr),
        compact_workspace.rho_cp_r,
        radial_cell_count,
        theta_cell_count,
        width_zone_count,
        static_cast<const double*>(cell_volumes_info.ptr)
    );
    double total_advection_time_s = 0.0;
    double total_diffusion_time_s = 0.0;
    int total_iterations = 0;

    for (int sub_idx = 0; sub_idx < std::max(substeps, 1); ++sub_idx) {
        const double t_sub = time_s + static_cast<double>(sub_idx) * dt_sub;
        if (enable_profiling) {
            const auto start = std::chrono::steady_clock::now();
            {
                py::gil_scoped_release release;
                advect_theta_periodic_core(
                    workspace.current.data(),
                    workspace.advected,
                    radial_cell_count,
                    theta_cell_count,
                    width_zone_count,
                    wheel_angular_speed_radps,
                    dt_sub,
                    theta_delta_rad
                );
            }
            total_advection_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } else {
            py::gil_scoped_release release;
            advect_theta_periodic_core(
                workspace.current.data(),
                workspace.advected,
                radial_cell_count,
                theta_cell_count,
                width_zone_count,
                wheel_angular_speed_radps,
                dt_sub,
                theta_delta_rad
            );
        }

        const auto& source_values = build_source_field_core_compressed(
            source_build_workspace,
            radial_cell_count,
            theta_cell_count,
            width_zone_count,
            source_volumetric_fraction,
            volumetric_source_w_per_m3,
            wheel_angular_speed_radps,
            t_sub,
            theta_delta_rad,
            static_cast<const long long*>(patch_radial_info.ptr),
            patch_radial_info.shape[0],
            static_cast<const long long*>(theta_offsets_info.ptr),
            theta_offsets_info.shape[0],
            static_cast<const long long*>(width_indices_info.ptr),
            width_indices_info.shape[0],
            compact_workspace.radial_layer_codes.data(),
            zone_weights,
            layer_source_weights,
            extra_source_ptr
        );
        const double source_power_w = source_power_w_native(
            source_values,
            radial_cell_count,
            theta_cell_count,
            width_zone_count,
            cell_volumes_m3
        );
        total_source_energy_j += dt_sub * source_power_w;

        std::pair<const std::vector<double>*, int> diffuse_result;
        if (enable_profiling) {
            const auto start = std::chrono::steady_clock::now();
            {
                py::gil_scoped_release release;
                diffuse_result = diffuse_vectorized_implicit_core_compact(
                    workspace.advected.data(),
                    source_values.data(),
                    compact_workspace.rho_cp_r,
                    compact_workspace.k_r_rw,
                    compact_workspace.k_theta_rw,
                    compact_workspace.k_w_rw,
                    dt_sub,
                    static_cast<const double*>(radial_minus_info.ptr),
                    static_cast<const double*>(radial_plus_info.ptr),
                    static_cast<const double*>(theta_coeff_info.ptr),
                    static_cast<const double*>(width_minus_info.ptr),
                    static_cast<const double*>(width_plus_info.ptr),
                    radial_cell_count,
                    theta_cell_count,
                    width_zone_count,
                    diffusion_max_iterations,
                    diffusion_tolerance_k,
                    parsed_solver_mode
                );
            }
            total_diffusion_time_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        } else {
            py::gil_scoped_release release;
            diffuse_result = diffuse_vectorized_implicit_core_compact(
                workspace.advected.data(),
                source_values.data(),
                compact_workspace.rho_cp_r,
                compact_workspace.k_r_rw,
                compact_workspace.k_theta_rw,
                compact_workspace.k_w_rw,
                dt_sub,
                static_cast<const double*>(radial_minus_info.ptr),
                static_cast<const double*>(radial_plus_info.ptr),
                static_cast<const double*>(theta_coeff_info.ptr),
                static_cast<const double*>(width_minus_info.ptr),
                static_cast<const double*>(width_plus_info.ptr),
                radial_cell_count,
                theta_cell_count,
                width_zone_count,
                diffusion_max_iterations,
                diffusion_tolerance_k,
                parsed_solver_mode
            );
        }

        total_iterations += diffuse_result.second;
        const auto* next_field_ptr = diffuse_result.first->data();
        for (ssize_t idx = 0; idx < item_count; ++idx) {
            workspace.current[static_cast<size_t>(idx)] = std::clamp(
                next_field_ptr[idx],
                minimum_temperature_k,
                maximum_temperature_k
            );
        }
    }
    const double final_energy_j = field_energy_j_native_compact(
        workspace.current.data(),
        compact_workspace.rho_cp_r,
        radial_cell_count,
        theta_cell_count,
        width_zone_count,
        static_cast<const double*>(cell_volumes_info.ptr)
    );

    return std::make_tuple(
        vector_to_array3d(workspace.current, radial_cell_count, theta_cell_count, width_zone_count),
        total_iterations,
        total_source_energy_j,
        initial_energy_j,
        final_energy_j,
        total_advection_time_s,
        total_diffusion_time_s
    );
}

}  // namespace

PYBIND11_MODULE(_hf_diffusion_native, m) {
    m.doc() = "Native high-fidelity diffusion kernel";
    m.def("diffuse_vectorized_implicit", &diffuse_vectorized_implicit);
    m.def("build_source_field", &build_source_field);
    m.def("build_source_and_diffuse_implicit", &build_source_and_diffuse_implicit);
    m.def("thermal_step_multi_substep", &thermal_step_multi_substep);
}
