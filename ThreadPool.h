#ifndef THREAD_POOL
#define THREAD_POOL

#include "Kernel.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

class MMF32Params;

struct MMF32FullParams {
  char *a;
  char *b;
  char *c;
  int blockSize;
  MMF32Params *params;
  char *T;
};

class ThreadPool {
 public:
  ThreadPool(int _numThreads = 0);
  ~ThreadPool();
  void QueueJob(char *a, char *b, char *c, int blockSize, MMF32Params *params, char *T);
  bool BusyDone();
  void WaitDone();
  void Stop();
  bool Terminated();
  int NumThreads();

 private:
  void ThreadLoop();

  int numThreads = 0;
  int leftCount = 0;
  bool terminate = false;
  std::mutex mutexLock;
  std::condition_variable cond;
  std::vector<std::thread> threads;
  std::queue<MMF32FullParams> jobs;
};

#endif

