// This file is generated. Do not edit.
// Generated on: 17.05.2024 08:55:11


#include "lib_tflite_micro/api/xcore_config.h"
#include "lib_nn/api/version.h"
#include "lib_tflite_micro/api/version.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_context.h"

// #define TFLMC_XCORE_PROFILE
// #define TFLMC_CONV2D_PROFILE
// #define TFLMC_PRINT_TENSORS
// #define TFLMC_PRINT_INPUT_TENSORS

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

// Check lib_nn and lib_tflite_micro versions
// NOTE: xformer version is saved for debugging purposes
// If lib_nn and lib_tflite_micro versions are as expected,
// then the xformer version doesn't matter as the model should execute
// If major version is zero, then minor versions must match
// Otherwise, major versions must match and binary minor version
// must be less or equal to runtime minor version
// Check if runtime lib_tflite_micro version matches with compiled version
static_assert((0 == 0 && lib_tflite_micro::major_version == 0 && 6 == lib_tflite_micro::minor_version) ||
              (0 == lib_tflite_micro::major_version) ||
              (6  < lib_tflite_micro::minor_version),
             "Model has been compiled with lib_tflite_micro version incompatible with runtime lib_tflite_micro version!");

// Check if runtime lib_nn version matches with compiled version
static_assert((0 == 0 && lib_nn::major_version == 0 && 3 == lib_nn::minor_version) ||
              (0 == lib_nn::major_version) ||
              (3  < lib_nn::minor_version),
             "Model has been compiled with lib_nn version incompatible with runtime lib_nn version!");



constexpr int kTensorArenaSize = 1112;
#ifndef SHARED_TENSOR_ARENA
namespace {
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
}
#else
extern uint8_t tensor_arena[];
#endif

namespace {
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_BROADCAST_TO,  OP_LAST
};

#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS) || defined(TFLMC_CONV2D_PROFILE)
const char *op_strs[] = {
"OP_BROADCAST_TO", };

#endif
#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS)
unsigned char checksum(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}

#endif

#ifdef TFLMC_XCORE_PROFILE
int op_times[OP_LAST];
int op_counts[OP_LAST];
int64_t op_times_summed;
int time_t0, time_t1;
#endif

TfLiteContext ctx{};

TFLMRegistration registrations[OP_LAST];

struct {
const TfArray<2, int> tensor_dimension0 = { 2, { 8,23 } };
const TfArray<1, float> quant0_scale = { 1, { 0.0078426999971270561, } };
const TfArray<1, int> quant0_zero = { 1, { -1 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int32_t tensor_data1[4] = { 
    1, 5, 8, 23, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 4 } };
const TfArray<4, int> tensor_dimension2 = { 4, { 1,5,8,23 } };
const TfArray<2, int> inputs0 = { 2, { 0,1 } };
const TfArray<1, int> outputs0 = { 1, { 2 } };
} g0;

TfLiteTensor tflTensors[] = 
{{ {(int32_t*)(tensor_arena + 928)},(TfLiteIntArray*)&g0.tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },184, kTfLiteArenaRw, false, },
{ {(int32_t*)g0.tensor_data1},(TfLiteIntArray*)&g0.tensor_dimension1, kTfLiteInt32, {kTfLiteNoQuantization, nullptr }, {0,0},16, kTfLiteMmapRo, false, },
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },920, kTfLiteArenaRw, false, },
};

TfLiteNode tflNodes[] = 
{{ (TfLiteIntArray*)&g0.inputs0, (TfLiteIntArray*)&g0.outputs0, (TfLiteIntArray*)&g0.inputs0, nullptr, 0, },
};

used_operators_e used_ops[] =
{OP_BROADCAST_TO, };


// Indices into tflTensors and tflNodes for subgraphs
size_t tflTensors_subgraph_index[] = {0, 3, };
size_t tflNodes_subgraph_index[] = {0, 1, };

// Variable tensors
size_t varTensors_index[] = {};

// Input/output tensors
static const int inTensorIndices[] = {
  0, 
};

static const int outTensorIndices[] = {
  2, 
};

// Indices into inTensors and outTensors for subgraphs
size_t inTensors_subgraph_index[] = {0, 1, };
size_t outTensors_subgraph_index[] = {0, 1, };

// Scratch buffer variables
int scratch_buffer_idx;
const int scratch_buffer_offsets[0] = {  };
tflite::MicroContext mc;
tflite::MicroGraph micro_graph;
size_t currentSubgraphIndex = 0;

// Xcore context and thread variables
xc_context_config_t xc_config;
// When using USE_DDR_FIX for enabling LPDDR support, only one thread can be used
#ifdef USE_DDR_FIX
static_assert((1 == 1),
             "Only one thread can be used when using USE_DDR_FIX! Please recompile with one thread!");
#endif
constexpr int kStackWordsPerThread = 256;
constexpr int threadsStackSizeInUint64 = 1 * kStackWordsPerThread/2;
// We use uint64_t for xcThreadsStack so that it is aligned to 8 bytes
uint64_t xcThreadsStack[threadsStackSizeInUint64];

// Persistent buffer ptr
// Initialized to the tail end of the tensor arena
uint8_t *persistentBufferPtr;
// Functions to be used as function pointers for TfLiteContext and MicroContext 
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  // Align to double word
  bytes = ((bytes + 7) / 8) * 8;
  persistentBufferPtr -= bytes;
  return persistentBufferPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[currentSubgraphIndex] + tensor_idx];
}

static TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext *context, size_t bytes,
                                       int *buffer_idx) {
  *buffer_idx = scratch_buffer_idx++;
  return kTfLiteOk;
};

static void *GetScratchBuffer(struct TfLiteContext *context,
                                       int buffer_idx) {
  return tensor_arena + scratch_buffer_offsets[buffer_idx];
}

static TfLiteTensor* mc_AllocateTempInputTensor(const TfLiteNode* node, int index) {
      if (node->inputs->data[index] < 0) {
        return nullptr;
      }
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->inputs->data[index]];
}

static TfLiteTensor* mc_AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      if (node->outputs->data[index] < 0) {
        return nullptr;
      }
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->outputs->data[index]];
}

static void mc_DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* mc_external_context() {
  return &xc_config;
}

static tflite::MicroGraph& mc_graph() {
  return micro_graph;
}

static int mg_NumSubgraphs(){
  return sizeof(tflTensors_subgraph_index)/sizeof(size_t) - 1;
}

static size_t mg_NumSubgraphInputs(int subgraph_idx){
  return inTensors_subgraph_index[subgraph_idx+1] - inTensors_subgraph_index[subgraph_idx];
}

static size_t mg_NumSubgraphOutputs(int subgraph_idx){
  return outTensors_subgraph_index[subgraph_idx+1] - outTensors_subgraph_index[subgraph_idx];
}

static TfLiteEvalTensor* mg_GetSubgraphInput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + inTensorIndices[inTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteEvalTensor* mg_GetSubgraphOutput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + outTensorIndices[outTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteStatus mg_InvokeSubgraph(int g){
  int prevSubgraphIndex = currentSubgraphIndex;
  currentSubgraphIndex = g;
#ifdef TFLMC_PRINT_TENSORS
printf("[\n");
#endif

  for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {

#ifdef TFLMC_PRINT_INPUT_TENSORS
    // print every input tensor
    printf("\nnode in %d", i);
    for (int j=0; j<tflNodes[i].inputs->size; j++){
      // -1 such as in case of no bias tensor for conv
      if (tflNodes[i].inputs->data[j] != -1) {
        printf("\ntensor %d, input %d, %d bytes, checksum %d\n", tflNodes[i].inputs->data[j], j, tflTensors[tflNodes[i].inputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].inputs->data[j]].data.raw, tflTensors[tflNodes[i].inputs->data[j]].bytes));
        for(int k=0; k<tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]].bytes; k++){
          printf("%d,", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]].data.raw[k]);
        }
      }
    }
    printf("\n");
#endif

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
  asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
  asm volatile ("gettime %0" : "=r" (time_t1));
#endif
  op_times[used_ops[i]] += time_t1 - time_t0;
  op_counts[used_ops[i]] += 1;
  printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

#ifdef TFLMC_PRINT_TENSORS
    // print every output tensor
    printf("\n{\"node\" : \"%d\", \"op\" : \"%s\", \"data\" : [", i, op_strs[used_ops[i]]);
    for (int j=0; j<tflNodes[i].outputs->size; j++){
      printf("\n{\"tensor\" : %d, \"output\" : %d, \"bytes\" : %d, \"checksum\" : %d,\n", tflNodes[i].outputs->data[j], j, tflTensors[tflNodes[i].outputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].outputs->data[j]].data.raw, tflTensors[tflNodes[i].outputs->data[j]].bytes));
      printf("\"val\" : [");
      for(int k=0; k<tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].bytes; k++){
        printf("%d", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].data.raw[k]);
        if (k < tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].bytes-1){
          printf(",");
        }
      }
      if(j<tflNodes[i].outputs->size-1){
        printf("]},\n");
      } else {
        printf("]}]\n");
      }
    }

    if(i < ((tflNodes_subgraph_index[g+1] - tflNodes_subgraph_index[g]) - 1)){
      printf("},\n");
    } else {
      printf("}\n");
    }
#endif

    if (status != kTfLiteOk) {
      currentSubgraphIndex = prevSubgraphIndex;
      return status;
    }
  }
#ifdef TFLMC_PRINT_TENSORS
printf("\n]");
#endif

  currentSubgraphIndex = prevSubgraphIndex;
  return kTfLiteOk;
}

} // namespace

TfLiteTensor* model_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

TfLiteTensor* model_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

#pragma stackfunction 1000
TfLiteStatus model_init(void *flash_data) {
  // Clear and initialize
  scratch_buffer_idx = 0;
  persistentBufferPtr = tensor_arena + kTensorArenaSize;

  // Set flash data in xcore context config
  xc_config.flash_data = flash_data;
  // Set thread count specified in the compiler
  xc_config.model_thread_count = 1;
  // Set thread info
  xc_config.thread_info.nstackwords = kStackWordsPerThread;
  xc_config.thread_info.stacks = &xcThreadsStack[threadsStackSizeInUint64 - 1];

  // Setup microcontext functions
  mc.AllocateTempInputTensor = &mc_AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &mc_AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &mc_DeallocateTempTfLiteTensor;
  mc.external_context = &mc_external_context;
  mc.graph = &mc_graph;

  micro_graph.NumSubgraphs = &mg_NumSubgraphs;
  micro_graph.NumSubgraphInputs = &mg_NumSubgraphInputs;
  micro_graph.NumSubgraphOutputs = &mg_NumSubgraphOutputs;
  micro_graph.GetSubgraphInput = &mg_GetSubgraphInput;
  micro_graph.GetSubgraphOutput = &mg_GetSubgraphOutput;
  micro_graph.InvokeSubgraph = &mg_InvokeSubgraph;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  
  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 3;
  registrations[OP_BROADCAST_TO] = tflite::Register_BROADCAST_TO();


  // Allocate persistent buffers for variable tensors
  for (int i = 0; i < 0; i++) {
    tflTensors[varTensors_index[i]].data.data = AllocatePersistentBuffer(&ctx, tflTensors[varTensors_index[i]].bytes);
  }

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling init()...");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

  for(size_t g = 0; g < 1; ++g) {
    currentSubgraphIndex = g;
    for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].init) {

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

      tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, tflNodes[i].custom_initial_data_size);

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t1));
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for init()...");
    for(int i=0; i<OP_LAST; i++){
      op_times_summed += op_times[i];
      printf("\n%-32s %-12d %.2fms", op_strs[i], op_times[i], op_times[i]/100000.0);
    }
    printf("\n\nTotal time for init() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
  printf("\n");
  printf("\nProfiling prepare()...");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

  for(size_t g = 0; g < 1; ++g) {
        currentSubgraphIndex = g;
        for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].prepare) {

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif
#endif

      TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
#ifdef __xcore__
      asm volatile ("gettime %0" : "=r" (time_t1));
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
printf("\n\nCumulative times for prepare()...");
    for(int i=0; i<OP_LAST; i++){
      op_times_summed += op_times[i];
      printf("\n%-32s %-12d %.2fms", op_strs[i], op_times[i], op_times[i]/100000.0);
    }
    printf("\n\nTotal time for prepare() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
  printf("\n");
#endif

  return kTfLiteOk;
}

#pragma stackfunction 1000
TfLiteStatus model_invoke() {
  thread_init_1(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling invoke()...");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
  op_times_summed = 0;
#endif

  mg_InvokeSubgraph(0);

  thread_destroy(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\n\nCumulative times for invoke()...");
  for(int i=0; i<OP_LAST; i++){
    op_times_summed += op_times[i];
    printf("\n%-5d %-32s %-12d %.2fms", op_counts[i], op_strs[i], op_times[i], op_times[i]/100000.0);
  }
  printf("\n\nTotal time for invoke() - %-10lld %.2fms\n\n", op_times_summed, op_times_summed/100000.0);
#endif

  return kTfLiteOk;
}

TfLiteStatus model_reset() {
  // Reset variable tensors
  for (int i = 0; i < 0; i++) {
    memset(tflTensors[varTensors_index[i]].data.data, tflTensors[varTensors_index[i]].params.zero_point, tflTensors[varTensors_index[i]].bytes);
  }
  return kTfLiteOk;
}

#if defined(__xcore__) && defined(USB_TILE)
#include "ioserver.h"
#include <xcore/hwtimer.h>
extern "C" {
extern int read_sswitch_reg(unsigned tile, unsigned reg, unsigned *data);
extern int write_sswitch_reg(unsigned tile, unsigned reg, unsigned data);
}

void model_ioserver(chanend_t c) {
    unsigned tensor_num = 0;
    extern unsigned tile[];
    while(1) {
        int cmd = ioserver_command_receive(c, &tensor_num);
        switch(cmd) {
        case IOSERVER_TENSOR_RECV_INPUT: {
            ioserver_tensor_recv_input(
                c, (unsigned int *) model_input(tensor_num)->data.u32,
                (model_input(tensor_num)->bytes + 3) / sizeof(int));
            break;
        }
        case IOSERVER_TENSOR_SEND_OUTPUT: {
            ioserver_tensor_send_output(
                c, (unsigned int*) model_output(tensor_num)->data.u32, 
                (model_output(tensor_num)->bytes + 3) / sizeof(int));
            break;
        }
        case IOSERVER_INVOKE: {
            model_invoke();
            ioserver_command_acknowledge(c, IOSERVER_ACK);
            break;
        }
        case IOSERVER_RESET: {
            model_reset();
            ioserver_command_acknowledge(c, IOSERVER_ACK);
            break;
        }
        case IOSERVER_EXIT: {
          ioserver_command_acknowledge(c, IOSERVER_ACK);
          unsigned pll_ctrl;
          hwtimer_t timer = hwtimer_alloc();
          hwtimer_delay(timer, 100000);
          hwtimer_free(timer);
          read_sswitch_reg(tile[USB_TILE], XS1_SSWITCH_PLL_CTL_NUM, &pll_ctrl);
          write_sswitch_reg(tile[USB_TILE], XS1_SSWITCH_PLL_CTL_NUM, pll_ctrl);
          return;
        }
        default: {
            ioserver_command_acknowledge(c, IOSERVER_NACK);
            break;
        }
        }
    }
}
#else 

void model_ioserver(void *io_channel) {}

#endif // __xcore__

