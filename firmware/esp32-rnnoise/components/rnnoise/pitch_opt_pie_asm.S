    .section .text
    .global xcorr_dot_product_pie_asm
    .type   xcorr_dot_product_pie_asm, @function
    .align  4

/*
 * int64_t xcorr_dot_product_pie_asm(const int16_t *x, const int16_t *y, int len);
 * 
 * HYBRID KERNEL - SIMD COMPONENT (Mult-Block Edition)
 * - Triple-Prime initialized history.
 * - Multiples of 8/24 in SIMD, remainders in C.
 */
xcorr_dot_product_pie_asm:
    entry   a1, 16

    # a2=X, a3=Y, a4=len
    
    # 1. Reset 40-bit Accumulator
    movi    a6, 0
    wur.accx_0 a6
    wur.accx_1 a6                

    # 2. Initialization: Triple-Prime
    ee.ld.128.usar.ip q1, a3, 16    # h1 = B0, a3 -> B1
    ee.ld.128.usar.ip q2, a3, 16    # h2 = B1, a3 -> B2
    ee.ld.128.usar.ip q3, a3, 16    # h3 = B2, a3 -> B3
    
    # Process blocks of 8
    srli    a6, a4, 3               
    
    # Unrolled Loop (Chunks of 24)
    movi    a5, 3
    quou    a5, a6, a5              
    
    beqz    a5, .L_loop_remainder

.L_unroll_24:
    ee.vld.128.ip q0, a2, 16
    ee.vmulas.s16.accx.ld.ip.qup q1, a3, 16, q0, q1, q2, q3
    
    ee.vld.128.ip q0, a2, 16
    ee.vmulas.s16.accx.ld.ip.qup q2, a3, 16, q0, q2, q3, q1
    
    ee.vld.128.ip q0, a2, 16
    ee.vmulas.s16.accx.ld.ip.qup q3, a3, 16, q0, q3, q1, q2

    addi    a5, a5, -1
    bnez    a5, .L_unroll_24

.L_loop_remainder:
    movi    a5, 3
    remu    a5, a6, a5
    beqz    a5, .L_finish

.L_rem_step:
    ee.vld.128.ip q0, a2, 16
    ee.vmulas.s16.accx.ld.ip.qup q1, a3, 16, q0, q1, q2, q3
    addi    a5, a5, -1
    beqz    a5, .L_finish
    
    ee.vld.128.ip q0, a2, 16
    ee.vmulas.s16.accx.ld.ip.qup q2, a3, 16, q0, q2, q3, q1

.L_finish:
    # Extraction
    rur.accx_0 a2               
    rur.accx_1 a3               
    slli    a3, a3, 24
    srai    a3, a3, 24
    retw
