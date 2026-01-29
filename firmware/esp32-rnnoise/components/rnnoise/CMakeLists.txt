idf_component_register(
    SRCS
        "denoise.c"
        "rnn.c"
        "nnet.c"
        "nnet_default.c"
        "pitch.c"
        "celt_lpc.c"
        "kiss_fft.c"
        "parse_lpcnet_weights.c"
        "rnnoise_data.c"
        "rnnoise_tables.c"
    INCLUDE_DIRS
        "."
    PRIV_REQUIRES
        esp_timer
        esp_common
        esp_system
        log
        freertos
    REQUIRES esp_mm esp_hw_support heap
)

# Define FLOAT_APPROX as a compile definition (not just a flag)
target_compile_definitions(${COMPONENT_LIB} PRIVATE
    FLOAT_APPROX=1
    OPUS_HAVE_RTCD=0
    CPU_INFO_BY_C=0
    OPUS_X86_MAY_HAVE_SSE=0
    OPUS_X86_MAY_HAVE_SSE2=0
    OPUS_X86_MAY_HAVE_SSE4_1=0
    OPUS_X86_MAY_HAVE_AVX=0
    DISABLE_NEON=1
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    # === Enable fast-math for AI/DSP ===
    -ffast-math                     # Enable fast-math (Flush-to-Zero for speed)
    -D__FAST_MATH__
    
    # === Optimization ===
    -O3                             # Aggressive optimization
    -funroll-loops                  # Loop unrolling
    
    # === Suppress warnings ===
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-sign-compare
)