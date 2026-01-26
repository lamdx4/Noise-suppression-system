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
        ""
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
    # === Disable fast-math (ESP-IDF may enable it by default) ===
    -fno-fast-math                  # Explicitly disable fast-math
    -U__FAST_MATH__                 # Undefine __FAST_MATH__ macro
    
    # === Optimization ===
    -O3                             # Aggressive optimization
    -funroll-loops                  # Loop unrolling
    
    # === Suppress warnings ===
    -Wno-unused-variable
    -Wno-unused-function
    -Wno-sign-compare
)
