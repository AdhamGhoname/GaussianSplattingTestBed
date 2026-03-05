#pragma once

#include "GaussianData.h"
#include <string>

/// Load a 3D Gaussian Splatting .ply file from disk.
/// Supports both binary_little_endian and ascii PLY formats.
/// Throws std::runtime_error on failure.
GaussianCloud loadPlyFile(const std::string& path);
