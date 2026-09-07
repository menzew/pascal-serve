// Exhaust every four-lane Q6 input against independent signed scalar subtraction.
#include <cstdint>
#include <cstdio>

int main() {
    for (uint32_t code = 0; code < (1u << 24); ++code) {
        uint32_t packed = 0;
        uint32_t reference = 0;
        for (int lane = 0; lane < 4; ++lane) {
            const int value = int((code >> (6 * lane)) & 63u);
            packed |= uint32_t(value) << (8 * lane);
            reference |= uint32_t(uint8_t(int8_t(value - 32))) << (8 * lane);
        }
        const uint32_t actual = ((packed | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
        if (actual != reference) {
            std::fprintf(stderr, "Mismatch at %u\n", code);
            return 1;
        }
    }
    std::puts("PASS: all 16,777,216 packed Q6 combinations");
}
