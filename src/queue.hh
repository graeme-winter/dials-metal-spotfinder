// A bounded blocking queue, used to hand work from the thread that dispatches
// frames to the threads that read and threshold them.
//
// Bounded is the point: if the threshold cannot keep up with the reads, push()
// blocks, rather than the backlog growing until the machine runs out of
// memory.

#ifndef SPOTFINDER_QUEUE_HH
#define SPOTFINDER_QUEUE_HH

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace work {

template <typename T> class Queue {
public:
  explicit Queue(std::size_t capacity) : capacity_(capacity) {}

  // Blocks while the queue is full. Returns false if the queue was closed.
  bool push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock,
                   [this] { return closed_ || items_.size() < capacity_; });
    if (closed_)
      return false;
    items_.push_back(std::move(value));
    if (items_.size() > high_water_)
      high_water_ = items_.size();
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Blocks while the queue is empty. Returns false once the queue is closed and
  // drained, which is how workers learn to stop.
  bool pop(T *value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
    if (items_.empty())
      return false;
    *value = std::move(items_.front());
    items_.pop_front();
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  // Wakes everybody. Items already queued are still handed out by pop().
  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  std::size_t high_water() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return high_water_;
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<T> items_;
  std::size_t capacity_;
  std::size_t high_water_ = 0;
  bool closed_ = false;
};

} // namespace work

#endif // SPOTFINDER_QUEUE_HH
