
#include "tst_common.h"

#include <xs1.h>
#include <xclib.h>
#include <stdio.h>
#include <assert.h>

#define CRC_POLY (0xEB31D82E)


int16_t  pseudo_rand_int16(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    return (int16_t)*r;
}

uint16_t pseudo_rand_uint16(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    return (uint16_t)*r;
}

int32_t  pseudo_rand_int32(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    return (int32_t)*r;
}

uint32_t pseudo_rand_uint32(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    return (uint32_t)*r;
}

int64_t  pseudo_rand_int64(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    int64_t a = (int64_t)*r;
    crc32(*r, -1, CRC_POLY);
    int64_t b = (int64_t)*r;
    return (int64_t)(a + (b<<32));
}

uint64_t pseudo_rand_uint64(unsigned *r){
    crc32(*r, -1, CRC_POLY);
    int64_t a = (int64_t)*r;
    crc32(*r, -1, CRC_POLY);
    int64_t b = (int64_t)*r;
    return (uint64_t)(a + (b<<32));
}

void pseudo_rand_bytes(unsigned *r, char* buffer, unsigned size){

    assert((((unsigned)buffer) & 0x3) == 0);

    unsigned b = 0;

    while(size >= sizeof(unsigned)){
        crc32(*r, -1, CRC_POLY);

        char* rch = (char*) r;

        for(int i = 0; i < sizeof(unsigned); i++)
            buffer[b++] = rch[i];

        size -= sizeof(unsigned);
    }
    
    crc32(*r, -1, CRC_POLY);
    unsigned tmp = *r;
    while(size){
        buffer[b++] = (char) (tmp & 0xFF);
        tmp >>= 8;
        size--;
    }
}


void print_warns(
    int start_case, 
    unsigned test_c, 
    unsigned test_asm)
{
    if(start_case != -1 && start_case != 0){
        printf("\t\t\t\t\t\t************************************************\n");
        printf("\t\t\t\t\t\t***** WARNING: Test not started on case 0 ******\n");
        printf("\t\t\t\t\t\t************************************************\n");
    }
    if(!test_c){
        printf("\t\t\t\t\t\t************************************************\n");
        printf("\t\t\t\t\t\t***** WARNING: Not testing C implementation ****\n");
        printf("\t\t\t\t\t\t************************************************\n");
    }
    if(!test_asm){
        printf("\t\t\t\t\t\t************************************************\n");
        printf("\t\t\t\t\t\t***** WARNING: Not testing ASM implementation **\n");
        printf("\t\t\t\t\t\t************************************************\n");
    }
}