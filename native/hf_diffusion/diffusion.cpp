#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

using Array3D = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Array1D = py::array_t<double, py::array::c_style | py::array::forcecast>;

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

    if (source_info.shape != field_info.shape || rho_cp_info.shape != field_info.shape ||
        k_r_info.shape != field_info.shape || k_theta_info.shape != field_info.shape ||
        k_w_info.shape != field_info.shape) {
        throw std::runtime_error("All 3D inputs must have identical shapes");
    }

    const ssize_t radial_cells = field_info.shape[0];
    const ssize_t theta_cells = field_info.shape[1];
    const ssize_t width_zones = field_info.shape[2];
    const ssize_t item_count = radial_cells * theta_cells * width_zones;

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

    const auto* field_ptr = static_cast<const double*>(field_info.ptr);
    const auto* source_ptr = static_cast<const double*>(source_info.ptr);
    const auto* rho_cp_ptr = static_cast<const double*>(rho_cp_info.ptr);
    const auto* k_r_ptr = static_cast<const double*>(k_r_info.ptr);
    const auto* k_theta_ptr = static_cast<const double*>(k_theta_info.ptr);
    const auto* k_w_ptr = static_cast<const double*>(k_w_info.ptr);
    const auto* radial_minus_ptr = static_cast<const double*>(radial_minus_info.ptr);
    const auto* radial_plus_ptr = static_cast<const double*>(radial_plus_info.ptr);
    const auto* theta_coeff_ptr = static_cast<const double*>(theta_coeff_info.ptr);
    const auto* width_minus_ptr = static_cast<const double*>(width_minus_info.ptr);
    const auto* width_plus_ptr = static_cast<const double*>(width_plus_info.ptr);

    auto index = [theta_cells, width_zones](ssize_t r, ssize_t t, ssize_t w) -> ssize_t {
        return ((r * theta_cells) + t) * width_zones + w;
    };

    std::vector<double> rhs(item_count);
    std::vector<double> estimate(item_count);
    std::vector<double> updated(item_count);
    std::vector<double> radial_prev(item_count, 0.0);
    std::vector<double> radial_next(item_count, 0.0);
    std::vector<double> width_prev(item_count, 0.0);
    std::vector<double> width_next(item_count, 0.0);
    std::vector<double> theta_prev(item_count, 0.0);
    std::vector<double> theta_next(item_count, 0.0);
    std::vector<double> coeff_r_minus(item_count);
    std::vector<double> coeff_r_plus(item_count);
    std::vector<double> coeff_theta(item_count);
    std::vector<double> coeff_w_minus(item_count);
    std::vector<double> coeff_w_plus(item_count);
    std::vector<double> diagonal(item_count);

    for (ssize_t r = 0; r < radial_cells; ++r) {
        for (ssize_t t = 0; t < theta_cells; ++t) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                const ssize_t idx = index(r, t, w);
                const double rho_safe = std::max(rho_cp_ptr[idx], 1e-12);
                const double alpha_r = k_r_ptr[idx] / rho_safe;
                const double alpha_theta = k_theta_ptr[idx] / rho_safe;
                const double alpha_w = k_w_ptr[idx] / rho_safe;
                rhs[idx] = field_ptr[idx] + dt_s * source_ptr[idx] / rho_safe;
                coeff_r_minus[idx] = dt_s * alpha_r * radial_minus_ptr[r];
                coeff_r_plus[idx] = dt_s * alpha_r * radial_plus_ptr[r];
                coeff_theta[idx] = dt_s * alpha_theta * theta_coeff_ptr[r];
                coeff_w_minus[idx] = dt_s * alpha_w * width_minus_ptr[w];
                coeff_w_plus[idx] = dt_s * alpha_w * width_plus_ptr[w];
                diagonal[idx] = 1.0 + coeff_r_minus[idx] + coeff_r_plus[idx] + 2.0 * coeff_theta[idx] +
                                coeff_w_minus[idx] + coeff_w_plus[idx];
                estimate[idx] = rhs[idx];
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
                    radial_prev[index(r, t, w)] = estimate[index(r - 1, t, w)];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells - 1; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 0; w < width_zones; ++w) {
                    radial_next[index(r, t, w)] = estimate[index(r + 1, t, w)];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t t = 0; t < theta_cells; ++t) {
                for (ssize_t w = 1; w < width_zones; ++w) {
                    width_prev[index(r, t, w)] = estimate[index(r, t, w - 1)];
                }
                for (ssize_t w = 0; w < width_zones - 1; ++w) {
                    width_next[index(r, t, w)] = estimate[index(r, t, w + 1)];
                }
            }
        }
        for (ssize_t r = 0; r < radial_cells; ++r) {
            for (ssize_t w = 0; w < width_zones; ++w) {
                theta_prev[index(r, 0, w)] = estimate[index(r, theta_cells - 1, w)];
                for (ssize_t t = 1; t < theta_cells; ++t) {
                    theta_prev[index(r, t, w)] = estimate[index(r, t - 1, w)];
                }
                theta_next[index(r, theta_cells - 1, w)] = estimate[index(r, 0, w)];
                for (ssize_t t = 0; t < theta_cells - 1; ++t) {
                    theta_next[index(r, t, w)] = estimate[index(r, t + 1, w)];
                }
            }
        }

        double max_delta = 0.0;
        for (ssize_t idx = 0; idx < item_count; ++idx) {
            updated[idx] = rhs[idx];
            updated[idx] += coeff_r_minus[idx] * radial_prev[idx];
            updated[idx] += coeff_r_plus[idx] * radial_next[idx];
            updated[idx] += coeff_theta[idx] * theta_prev[idx];
            updated[idx] += coeff_theta[idx] * theta_next[idx];
            updated[idx] += coeff_w_minus[idx] * width_prev[idx];
            updated[idx] += coeff_w_plus[idx] * width_next[idx];
            updated[idx] /= std::max(diagonal[idx], 1e-12);
            max_delta = std::max(max_delta, std::abs(updated[idx] - estimate[idx]));
        }

        estimate.swap(updated);
        if (max_delta < diffusion_tolerance_k) {
            break;
        }
    }

    py::array_t<double> result({radial_cells, theta_cells, width_zones});
    auto result_info = result.request();
    auto* result_ptr = static_cast<double*>(result_info.ptr);
    std::copy(estimate.begin(), estimate.end(), result_ptr);
    return {result, iterations};
}

}  // namespace

PYBIND11_MODULE(_hf_diffusion_native, m) {
    m.doc() = "Native high-fidelity diffusion kernel";
    m.def("diffuse_vectorized_implicit", &diffuse_vectorized_implicit);
}
