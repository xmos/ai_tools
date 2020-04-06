
#include "tst_common.h"

#include <xs1.h>
#include <xclib.h>
#include <stdio.h>
#include <assert.h>

#define CRC_POLY (0xEB31D82E)

unsafe {
    void test_crc32(unsigned * unsafe r)
    {
        crc32(*r, -1, CRC_POLY);
    }
}