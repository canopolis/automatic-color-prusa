// Prusa Colorin / ColorMix portable: minimal shim of Slic3r Color.hpp.
// Only what the v0.9.18 colour engine uses; behaviour copied from Color.cpp.
#ifndef colorin_shim_Color_hpp_
#define colorin_shim_Color_hpp_
#include <array>
#include <string>
namespace Slic3r {
using RGBA = std::array<float, 4>;
class ColorRGB {
public:
    ColorRGB() = default;
    ColorRGB(float r, float g, float b) : m{r, g, b} {}
    float r() const { return m[0]; }
    float g() const { return m[1]; }
    float b() const { return m[2]; }
private:
    std::array<float, 3> m{0.f, 0.f, 0.f};
};
// Same rules as Slic3r::decode_color for "#RRGGBB" (and "#RRGGBBAA"):
// unknown digits leave the channel black, anything else returns false.
inline bool decode_color(const std::string& in, ColorRGB& out)
{
    auto h = [](char c) { return (c >= '0' && c <= '9') ? int(c - '0') : (c >= 'A' && c <= 'F') ? int(c - 'A') + 10 : (c >= 'a' && c <= 'f') ? int(c - 'a') + 10 : -1; };
    out = ColorRGB();
    if (in.empty() || in[0] != '#' || (in.size() != 7 && in.size() != 9)) return false;
    float ch[3] = {0.f, 0.f, 0.f};
    for (int i = 0; i < 3; ++i) {
        const int d1 = h(in[1 + 2 * i]), d2 = h(in[2 + 2 * i]);
        if (d1 != -1 && d2 != -1) ch[i] = float(d1 * 16 + d2) * (1.f / 255.f);
    }
    out = ColorRGB(ch[0], ch[1], ch[2]);
    return true;
}
} // namespace Slic3r
#endif
