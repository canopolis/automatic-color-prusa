// ColorMix portable - motor ratdoux (Snapmaker Orca FullSpectrum v0.9.18).
// C entry point around the unmodified v0.9.18 colour prediction:
// same steps as predict_candidate_color() in ImageMap/ContinuousColorSolver.cpp
// (one filament -> its own colour; several -> FullSpectrum KM/K-S spectral
// model; malformed colours -> filament_mixer fallback).
#include "Color.hpp"
#include "FullSpectrumKSPairResidual.hpp"
#include "filament_mixer_model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace Slic3r;

static std::vector<std::string> split(const char* s)
{
    std::vector<std::string> out;
    std::stringstream ss(s ? s : "");
    std::string item;
    while (std::getline(ss, item, ','))
        out.push_back(item);
    return out;
}

static int pack(float r, float g, float b)
{
    auto byte = [](float v) { return int(std::lround(std::clamp(std::isfinite(v) ? v : 0.f, 0.f, 1.f) * 255.f)); };
    return (byte(r) << 16) | (byte(g) << 8) | byte(b);
}

extern "C" {

// hex_list: "#RRGGBB,#RRGGBB,..."; unit_list: "12,8,0,..." (same order).
// Returns 0xRRGGBB, or -1 when nothing is active.
int cm_predict(const char* hex_list, const char* unit_list)
{
    const std::vector<std::string> hexes = split(hex_list);
    const std::vector<std::string> units_s = split(unit_list);
    std::vector<std::string> act_hex;
    std::vector<int>         act_units;
    for (size_t i = 0; i < hexes.size() && i < units_s.size(); ++i) {
        const int u = std::atoi(units_s[i].c_str());
        if (u <= 0) continue;
        act_hex.push_back(hexes[i]);
        act_units.push_back(u);
    }
    if (act_hex.empty())
        return -1;
    if (act_hex.size() == 1) {
        ColorRGB c;
        decode_color(act_hex.front(), c);
        return pack(c.r(), c.g(), c.b());
    }
    std::vector<FullSpectrumKSPairResidualColorInput> in;
    for (size_t i = 0; i < act_hex.size(); ++i)
        in.push_back({act_hex[i], act_units[i], std::nullopt, std::nullopt});
    if (const std::optional<std::string> p = full_spectrum_ks_blend_color_multi(in)) {
        ColorRGB c;
        if (decode_color(*p, c))
            return pack(c.r(), c.g(), c.b());
    }
    // v0.9.18 fallback: sequential filament_mixer lerp.
    auto byte = [](float v) { return (unsigned char)std::lround(std::clamp(v, 0.f, 1.f) * 255.f); };
    ColorRGB c0;
    decode_color(act_hex[0], c0);
    unsigned char r = byte(c0.r()), g = byte(c0.g()), b = byte(c0.b());
    int acc = act_units[0];
    for (size_t i = 1; i < act_hex.size(); ++i) {
        ColorRGB ci;
        decode_color(act_hex[i], ci);
        const int total = acc + act_units[i];
        filament_mixer::lerp(r, g, b, byte(ci.r()), byte(ci.g()), byte(ci.b()), float(act_units[i]) / float(total), &r, &g, &b);
        acc = total;
    }
    return (int(r) << 16) | (int(g) << 8) | int(b);
}

} // extern "C"
