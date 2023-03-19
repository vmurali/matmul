#include "Kernel.h"
#include "ThreadPool.h"

ThreadPool::ThreadPool(int _numThreads) : numThreads(_numThreads) {
  if (numThreads == 0)
    numThreads = std::thread::hardware_concurrency()-1;
  threads.reserve(numThreads);
  for (int i = 0; i < numThreads; i++) {
      threads.push_back(std::thread(&ThreadPool::ThreadLoop, this));
  }
}

int ThreadPool::NumThreads() {
  return numThreads;
}

void ThreadPool::ThreadLoop() {
  while (true) {
    MMF32FullParams job;
    {
      std::unique_lock<std::mutex> lock(mutexLock);
      cond.wait(lock, [this]() {
          return !jobs.empty() || terminate;
      });
      if (terminate) {
        return;
      }
      job = jobs.front();
      jobs.pop();
    }
    MMF32(job.a, job.b, job.c, job.blockSize, job.params, job.T);
    {
      std::unique_lock<std::mutex> lock(mutexLock);
      leftCount--;
      cond.notify_all();
    }
  }
}

bool ThreadPool::BusyDone() {
  bool done;
  {
    std::unique_lock<std::mutex> lock(mutexLock);
    done = leftCount == 0;
  }
  return done;
}

void ThreadPool::WaitDone() {
  std::unique_lock<std::mutex> lock(mutexLock);
  cond.wait(lock, [this]() {
    return leftCount == 0;
  });
}

void ThreadPool::QueueJob(char *a, char *b, char *c, int blockSize,
                          MMF32Params *params, char *T) {
  {
    std::unique_lock<std::mutex> lock(mutexLock);
    jobs.push(MMF32FullParams{a, b, c, blockSize, params, T});
    leftCount++;
  }
  cond.notify_one();
}

bool ThreadPool::Terminated() {
  std::unique_lock<std::mutex> lock(mutexLock);
  return terminate;
}

ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutexLock);
    terminate = true;
  }
  cond.notify_all();
  for (auto& th : threads) {
    th.join();
  }
}


