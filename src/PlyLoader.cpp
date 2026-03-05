#include "PlyLoader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

// ═════════════════════════════════════════════════════════════════════════════
//  PLY header parsing helpers
// ═════════════════════════════════════════════════════════════════════════════

enum class PlyFormat { Ascii, BinaryLittleEndian, BinaryBigEndian };

enum class PlyType {
    Float, Double,
    UChar, Short, UShort,
    Int, UInt,
    Char,
    Unknown
};

static PlyType parsePlyType(const std::string& s) {
    if (s == "float"  || s == "float32") return PlyType::Float;
    if (s == "double" || s == "float64") return PlyType::Double;
    if (s == "uchar"  || s == "uint8")   return PlyType::UChar;
    if (s == "short"  || s == "int16")   return PlyType::Short;
    if (s == "ushort" || s == "uint16")  return PlyType::UShort;
    if (s == "int"    || s == "int32")   return PlyType::Int;
    if (s == "uint"   || s == "uint32")  return PlyType::UInt;
    if (s == "char"   || s == "int8")    return PlyType::Char;
    return PlyType::Unknown;
}

static size_t plyTypeSize(PlyType t) {
    switch (t) {
        case PlyType::Char:   case PlyType::UChar:  return 1;
        case PlyType::Short:  case PlyType::UShort: return 2;
        case PlyType::Int:    case PlyType::UInt:
        case PlyType::Float:                        return 4;
        case PlyType::Double:                       return 8;
        default: return 0;
    }
}

struct PlyProperty {
    std::string name;
    PlyType     type     = PlyType::Unknown;
    bool        isList   = false;
    PlyType     countType = PlyType::Unknown; // only for list properties
    size_t      offset   = 0;                 // byte offset within the vertex row
};

struct PlyElement {
    std::string               name;
    uint32_t                  count = 0;
    std::vector<PlyProperty>  properties;
    size_t                    rowBytes = 0; // total bytes per row
};

// ─── read a float from any PLY numeric type ─────────────────────────────────
static float readFloatBinary(const uint8_t* data, PlyType type) {
    switch (type) {
        case PlyType::Float: {
            float v; memcpy(&v, data, 4); return v;
        }
        case PlyType::Double: {
            double v; memcpy(&v, data, 8); return static_cast<float>(v);
        }
        case PlyType::UChar:  return static_cast<float>(data[0]);
        case PlyType::Char:   return static_cast<float>(reinterpret_cast<const int8_t*>(data)[0]);
        case PlyType::Short: {
            int16_t v; memcpy(&v, data, 2); return static_cast<float>(v);
        }
        case PlyType::UShort: {
            uint16_t v; memcpy(&v, data, 2); return static_cast<float>(v);
        }
        case PlyType::Int: {
            int32_t v; memcpy(&v, data, 4); return static_cast<float>(v);
        }
        case PlyType::UInt: {
            uint32_t v; memcpy(&v, data, 4); return static_cast<float>(v);
        }
        default: return 0.0f;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Main loader
// ═════════════════════════════════════════════════════════════════════════════

GaussianCloud loadPlyFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("PLY: failed to open file: " + path);

    // ── parse header ─────────────────────────────────────────────────────
    std::string line;
    std::getline(file, line);
    if (line.find("ply") == std::string::npos)
        throw std::runtime_error("PLY: not a valid PLY file: " + path);

    PlyFormat format = PlyFormat::BinaryLittleEndian;
    std::vector<PlyElement> elements;
    PlyElement* currentElement = nullptr;

    while (std::getline(file, line)) {
        // Trim \r if present (Windows line endings)
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "end_header") break;

        if (token == "format") {
            std::string fmt;
            iss >> fmt;
            if (fmt == "binary_little_endian") format = PlyFormat::BinaryLittleEndian;
            else if (fmt == "binary_big_endian") format = PlyFormat::BinaryBigEndian;
            else if (fmt == "ascii") format = PlyFormat::Ascii;
            else throw std::runtime_error("PLY: unsupported format: " + fmt);
        }
        else if (token == "element") {
            PlyElement elem;
            iss >> elem.name >> elem.count;
            elements.push_back(std::move(elem));
            currentElement = &elements.back();
        }
        else if (token == "property" && currentElement) {
            std::string next;
            iss >> next;

            PlyProperty prop;
            if (next == "list") {
                prop.isList = true;
                std::string countTypeStr, elemTypeStr;
                iss >> countTypeStr >> elemTypeStr >> prop.name;
                prop.countType = parsePlyType(countTypeStr);
                prop.type      = parsePlyType(elemTypeStr);
            } else {
                prop.type = parsePlyType(next);
                iss >> prop.name;
            }
            currentElement->properties.push_back(std::move(prop));
        }
    }

    // Compute byte offsets for binary reading
    for (auto& elem : elements) {
        size_t offset = 0;
        for (auto& prop : elem.properties) {
            prop.offset = offset;
            offset += plyTypeSize(prop.type);
        }
        elem.rowBytes = offset;
    }

    // ── find the "vertex" element ────────────────────────────────────────
    PlyElement* vertexElem = nullptr;
    for (auto& elem : elements) {
        if (elem.name == "vertex") { vertexElem = &elem; break; }
    }
    if (!vertexElem)
        throw std::runtime_error("PLY: no 'vertex' element found in: " + path);

    // Build a name → property-index map
    std::unordered_map<std::string, size_t> propIndex;
    for (size_t i = 0; i < vertexElem->properties.size(); ++i)
        propIndex[vertexElem->properties[i].name] = i;

    // Helper: get property index or SIZE_MAX if absent
    auto propIdx = [&](const std::string& name) -> size_t {
        auto it = propIndex.find(name);
        return it != propIndex.end() ? it->second : SIZE_MAX;
    };

    // ── detect SH degree from properties ─────────────────────────────────
    // Count how many f_rest_N properties exist.
    uint32_t shRestCount = 0;
    while (propIndex.count("f_rest_" + std::to_string(shRestCount)))
        ++shRestCount;

    uint32_t shDegree = 0;
    uint32_t totalSHCoeffs = 3 + shRestCount; // DC (3) + rest
    if      (totalSHCoeffs >= 48) shDegree = 3;
    else if (totalSHCoeffs >= 27) shDegree = 2; // (deg+1)^2 * 3
    else if (totalSHCoeffs >= 12) shDegree = 1;

    std::cout << "PLY: " << vertexElem->count << " splats, SH degree " << shDegree
              << " (" << totalSHCoeffs << " SH coefficients)\n";

    // ── required property indices ────────────────────────────────────────
    size_t idx_x  = propIdx("x"),  idx_y  = propIdx("y"),  idx_z  = propIdx("z");
    size_t idx_nx = propIdx("nx"), idx_ny = propIdx("ny"), idx_nz = propIdx("nz");
    size_t idx_opacity = propIdx("opacity");

    size_t idx_f_dc[3]   = { propIdx("f_dc_0"), propIdx("f_dc_1"), propIdx("f_dc_2") };
    size_t idx_scale[3]  = { propIdx("scale_0"), propIdx("scale_1"), propIdx("scale_2") };
    size_t idx_rot[4]    = { propIdx("rot_0"), propIdx("rot_1"), propIdx("rot_2"), propIdx("rot_3") };

    std::vector<size_t> idx_f_rest(shRestCount);
    for (uint32_t i = 0; i < shRestCount; ++i)
        idx_f_rest[i] = propIdx("f_rest_" + std::to_string(i));

    if (idx_x == SIZE_MAX || idx_y == SIZE_MAX || idx_z == SIZE_MAX)
        throw std::runtime_error("PLY: missing position properties (x, y, z)");
    if (idx_opacity == SIZE_MAX)
        throw std::runtime_error("PLY: missing opacity property");

    // ── read vertex data ─────────────────────────────────────────────────
    GaussianCloud cloud;
    cloud.shDegree = shDegree;
    cloud.splats.resize(vertexElem->count);

    auto readProp = [&](const uint8_t* row, size_t pidx) -> float {
        if (pidx == SIZE_MAX) return 0.0f;
        const auto& prop = vertexElem->properties[pidx];
        return readFloatBinary(row + prop.offset, prop.type);
    };

    if (format == PlyFormat::BinaryLittleEndian || format == PlyFormat::BinaryBigEndian) {
        if (format == PlyFormat::BinaryBigEndian)
            std::cerr << "PLY: warning — big-endian not byte-swapped, results may be wrong\n";

        // Skip non-vertex elements that appear before "vertex"
        for (auto& elem : elements) {
            if (&elem == vertexElem) break;

            // Skip this element's data
            if (elem.properties.empty()) continue;
            bool hasList = false;
            for (auto& p : elem.properties) if (p.isList) { hasList = true; break; }

            if (!hasList) {
                file.seekg(static_cast<std::streamoff>(elem.rowBytes * elem.count), std::ios::cur);
            } else {
                // List properties: must read row by row
                for (uint32_t r = 0; r < elem.count; ++r) {
                    for (auto& p : elem.properties) {
                        if (p.isList) {
                            uint32_t listCount = 0;
                            size_t cs = plyTypeSize(p.countType);
                            file.read(reinterpret_cast<char*>(&listCount), cs);
                            file.seekg(static_cast<std::streamoff>(listCount * plyTypeSize(p.type)), std::ios::cur);
                        } else {
                            file.seekg(static_cast<std::streamoff>(plyTypeSize(p.type)), std::ios::cur);
                        }
                    }
                }
            }
        }

        // Read vertex rows
        std::vector<uint8_t> rowBuf(vertexElem->rowBytes);
        for (uint32_t i = 0; i < vertexElem->count; ++i) {
            file.read(reinterpret_cast<char*>(rowBuf.data()), vertexElem->rowBytes);
            if (!file)
                throw std::runtime_error("PLY: unexpected EOF while reading vertex " + std::to_string(i));

            auto& s = cloud.splats[i];
            const uint8_t* row = rowBuf.data();

            s.x  = readProp(row, idx_x);
            s.y  = readProp(row, idx_y);
            s.z  = readProp(row, idx_z);
            s.nx = readProp(row, idx_nx);
            s.ny = readProp(row, idx_ny);
            s.nz = readProp(row, idx_nz);

            for (int c = 0; c < 3; ++c)
                s.sh[c] = readProp(row, idx_f_dc[c]);

            for (uint32_t c = 0; c < shRestCount && c < 45; ++c)
                s.sh[3 + c] = readProp(row, idx_f_rest[c]);

            // Zero-fill unused SH coefficients
            for (uint32_t c = 3 + shRestCount; c < 48; ++c)
                s.sh[c] = 0.0f;

            s.opacity = readProp(row, idx_opacity);

            for (int c = 0; c < 3; ++c)
                s.scale[c] = readProp(row, idx_scale[c]);

            for (int c = 0; c < 4; ++c)
                s.rot[c] = readProp(row, idx_rot[c]);
        }
    }
    else if (format == PlyFormat::Ascii) {
        // Skip non-vertex elements before "vertex"
        for (auto& elem : elements) {
            if (&elem == vertexElem) break;
            for (uint32_t r = 0; r < elem.count; ++r)
                std::getline(file, line);
        }

        for (uint32_t i = 0; i < vertexElem->count; ++i) {
            if (!std::getline(file, line))
                throw std::runtime_error("PLY: unexpected EOF reading ASCII vertex " + std::to_string(i));

            std::istringstream iss(line);
            std::vector<float> vals(vertexElem->properties.size());
            for (size_t p = 0; p < vertexElem->properties.size(); ++p)
                iss >> vals[p];

            auto val = [&](size_t pidx) -> float {
                return pidx < vals.size() ? vals[pidx] : 0.0f;
            };

            auto& s = cloud.splats[i];
            s.x  = val(idx_x);   s.y  = val(idx_y);   s.z  = val(idx_z);
            s.nx = val(idx_nx);  s.ny = val(idx_ny);  s.nz = val(idx_nz);

            for (int c = 0; c < 3; ++c)
                s.sh[c] = val(idx_f_dc[c]);
            for (uint32_t c = 0; c < shRestCount && c < 45; ++c)
                s.sh[3 + c] = val(idx_f_rest[c]);
            for (uint32_t c = 3 + shRestCount; c < 48; ++c)
                s.sh[c] = 0.0f;

            s.opacity = val(idx_opacity);
            for (int c = 0; c < 3; ++c) s.scale[c] = val(idx_scale[c]);
            for (int c = 0; c < 4; ++c) s.rot[c]   = val(idx_rot[c]);
        }
    }

    std::cout << "PLY: loaded " << cloud.splats.size() << " Gaussian splats from " << path << "\n";
    return cloud;
}
