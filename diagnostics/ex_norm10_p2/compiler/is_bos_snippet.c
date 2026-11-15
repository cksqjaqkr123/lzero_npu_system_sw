float rmsnorm_quant(float x, float s_in, float s_out, bool is_bos, int16_t* lut) {
    if (is_bos) return rmsnorm_fp(x);
    int code = (int)roundf(x / s_in);
    code = code < -511 ? -511 : (code > 511 ? 511 : code);
    return (float)lut[code + 511] * s_out;
}
float gelu_quant(float gate, float s_in, float s_out, int16_t* lut) {
    int code = (int)roundf(gate / s_in);
    code = code < -511 ? -511 : (code > 511 ? 511 : code);
    return (float)lut[code + 511] * s_out;
}
