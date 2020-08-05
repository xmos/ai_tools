
#include "tst_common.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

int8_t  pseudo_rand_int8(){
    return (int8_t)rand();
}

int16_t  pseudo_rand_int16(){
    return (int16_t)rand();
}

uint16_t pseudo_rand_uint16(){
    return (uint16_t)rand();
}

int32_t  pseudo_rand_int32(){
    return (int32_t)rand();
}

uint32_t pseudo_rand_uint32(){
    return (uint32_t)rand();
}

int64_t  pseudo_rand_int64(){
    
    int64_t a = rand();
    int64_t b = rand();
    return (int64_t)(a + (b<<32));
}

uint64_t pseudo_rand_uint64(){
    int64_t a = (int64_t) rand();
    int64_t b = (int64_t) rand();
    return (uint64_t)(a + (b<<32));
}

void pseudo_rand_bytes(char* buffer, unsigned size){

    unsigned b = 0;

    while(size >= sizeof(unsigned)){

        int r = rand();

        char* rch = (char*) &r;

        for(int i = 0; i < sizeof(unsigned); i++)
            buffer[b++] = rch[i];

        size -= sizeof(unsigned);
    }
    
    unsigned tmp = rand();
    while(size){
        buffer[b++] = (char) (tmp & 0xFF);
        tmp >>= 8;
        size--;
    }
}


void print_warns(
    int start_case)
{
    if(start_case != -1 && start_case != 0){
        printf("\t\t\t\t\t\t************************************************\n");
        printf("\t\t\t\t\t\t***** WARNING: Test not started on case 0 ******\n");
        printf("\t\t\t\t\t\t************************************************\n");
    }
}