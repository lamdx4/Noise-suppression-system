/* Simple stub for os_support.h - ESP32 PORT */
#ifndef OS_SUPPORT_H
#define OS_SUPPORT_H

#include <string.h>

/* OPUS_CLEAR macro */
#define OPUS_CLEAR(dst, n) (memset((dst), 0, (n)*sizeof(*(dst))))

#endif /* OS_SUPPORT_H */
