// Copyright (C) Mihai Preda

#include "Kernel.h"
#include "KernelCompiler.h"
#include "log.h"

#include <stdexcept>

// Work-group info queries used by logWorkGroupInfo() (not in tinycl.h/tinycuda.h).
#ifndef CL_KERNEL_WORK_GROUP_SIZE
#define CL_KERNEL_WORK_GROUP_SIZE                    0x11B0
#endif
#ifndef CL_KERNEL_LOCAL_MEM_SIZE
#define CL_KERNEL_LOCAL_MEM_SIZE                     0x11B2
#endif
#ifndef CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
#define CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#endif
#ifndef CL_KERNEL_PRIVATE_MEM_SIZE
#define CL_KERNEL_PRIVATE_MEM_SIZE                   0x11B4
#endif

Kernel::Kernel(string_view name, KernelCompiler* compiler, TimeInfo* timeInfo, Queue* queue,
       string_view fileName, string_view nameInFile,
       size_t workSize, string_view defines):
  name{name},
  compiler{compiler},
  fileName{fileName},
  nameInFile{nameInFile},
  defines{defines},
  timeInfo{timeInfo},
  queue{queue},
  workSize{workSize}
{}

Kernel::~Kernel() = default;

void Kernel::startLoad(KernelCompiler* compiler) {
  assert(!kernel);
  assert(!pendingKernel.valid());
  pendingKernel = compiler->load(fileName, nameInFile, defines);
  deviceId = compiler->deviceId;
}

void Kernel::finishLoad() {
  pendingKernel.wait();
  kernel = pendingKernel.get();
  assert(kernel);
  groupSize = getWorkGroupSize(kernel.get(), deviceId, name.c_str());
  assert(groupSize);
  assert(workSize % groupSize == 0);

  for (auto [pos, arg] : pendingArgs) { setArgs(pos, arg); }
}

void Kernel::run() {
  assert(kernel);
  queue->run(kernel.get(), groupSize, workSize, timeInfo);
}

// Diagnostic introspection (megapass MEGA_PROF/MEGA_DEBUG). Compiles the kernel if it is
// not loaded yet (exactly the lazy-load path operator() uses) but NEVER enqueues it, then
// logs, per kernel:
//   maxWG    = CL_KERNEL_WORK_GROUP_SIZE                    (reg-limited max threads/WG -- the occupancy signal)
//   privMem  = CL_KERNEL_PRIVATE_MEM_SIZE  (bytes/thread)
//   locMem   = CL_KERNEL_LOCAL_MEM_SIZE    (bytes/WG)
//   prefMult = CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
// A query the driver doesn't support logs as 0 rather than failing.
void Kernel::logWorkGroupInfo() {
  if (!kernel) {
    startLoad(compiler);
    finishLoad();
  }
  if (!kernel) {
    log("MEGAPROF: %s: kernel not found\n", name.c_str());
    return;
  }

  size_t maxWG = 0, prefMult = 0;
  u64 privMem = 0, locMem = 0;   // cl_ulong results
  (void) clGetKernelWorkGroupInfo(kernel.get(), deviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(maxWG), &maxWG, NULL);
  (void) clGetKernelWorkGroupInfo(kernel.get(), deviceId, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(privMem), &privMem, NULL);
  (void) clGetKernelWorkGroupInfo(kernel.get(), deviceId, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(locMem), &locMem, NULL);
  (void) clGetKernelWorkGroupInfo(kernel.get(), deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(prefMult), &prefMult, NULL);

  log("MEGAPROF: %-18s maxWG=%4u privMem=%4u locMem=%5u prefMult=%u\n",
      name.c_str(), u32(maxWG), u32(privMem), u32(locMem), u32(prefMult));
}
