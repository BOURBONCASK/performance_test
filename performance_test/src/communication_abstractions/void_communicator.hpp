// Copyright 2017 Apex.AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef COMMUNICATION_ABSTRACTIONS__VOID_COMMUNICATOR_HPP_
#define COMMUNICATION_ABSTRACTIONS__VOID_COMMUNICATOR_HPP_

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

namespace performance_test
{

/**
 * \brief Communication plugin for Non or VOID middleware.
 * \tparam Msg topic type to use.
 */
template<class Msg>
class VoidCommunicator : public Communicator
{
public:

  /// The data type to publish and subscribe to. Using ROS types.
  using DataType = typename Msg::RosType;

  /// Constructor which takes a reference \param lock to the lock to use.
  explicit VoidCommunicator(DataStats & stats)
  : Communicator(stats)
  {
  }

  /**
   * \brief Publishes the provided data.
   *
   * \param time The time to fill into the data field.
   */
  void publish(std::int64_t time)
  {
    if (!m_ec.is_zero_copy_transfer()) {
      // Creating a new data type to measure its creation time
      DataType useless_data;
      static_cast<void>(useless_data);
    }

    m_stats.lock();

    VoidCommunicator<Msg>::m_data_shared_.push(std::make_pair(time, m_stats.next_sample_id()));
    {
      std::lock_guard<std::mutex> lk(VoidCommunicator<Msg>::cv_mutex_);
      VoidCommunicator<Msg>::data_to_read_++;
    }

    m_stats.unlock();

    VoidCommunicator<Msg>::cv_.notify_one();
  }
  /**
   * \brief Reads received data from shared vector.
   */
  void update_subscription()
  {
    std::unique_lock<std::mutex> lk(VoidCommunicator<Msg>::cv_mutex_);
    cv_.wait(lk, []{return VoidCommunicator<Msg>::data_to_read_ > 0;});

    m_stats.lock();

    std::pair<int64_t, uint64_t> data = m_data_shared_.front();
    m_data_shared_.pop();

    m_stats.update_subscriber_stats(data.first, data.second, sizeof(DataType));

    m_stats.increment_received();
    {
      VoidCommunicator<Msg>::data_to_read_--;
    }

    m_stats.unlock();
  }

  /*
  /// Returns the data received in bytes.
  std::size_t data_received()
  {
    return m_stats.num_received_samples() * sizeof(DataType);
  }
  */

private:
  static std::queue<std::pair<int64_t, uint64_t>> m_data_shared_;
  static std::condition_variable cv_;
  static std::mutex cv_mutex_;
  static std::atomic<int> data_to_read_;
};

template<class Msg>
std::queue<std::pair<int64_t, uint64_t>> VoidCommunicator<Msg>::m_data_shared_;

template<class Msg>
std::condition_variable VoidCommunicator<Msg>::cv_;

template<class Msg>
std::mutex VoidCommunicator<Msg>::cv_mutex_;

template<class Msg>
std::atomic<int> VoidCommunicator<Msg>::data_to_read_;

}  // namespace performance_test

#endif  // COMMUNICATION_ABSTRACTIONS__VOID_COMMUNICATOR_HPP_
