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

#include "performance_test/communication_abstractions/communicator_factory.hpp"

#include <memory>
#include <string>

#include "performance_test/for_each.hpp"
#include "performance_test/generated_messages/messages.hpp"


#ifdef PERFORMANCE_TEST_RCLCPP_ENABLED
#include "performance_test/communication_abstractions/rclcpp_publisher.hpp"
#endif

#if defined(PERFORMANCE_TEST_RCLCPP_STE_ENABLED) || defined(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED)
#include "performance_test/communication_abstractions/rclcpp_callback_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED
#include "performance_test/communication_abstractions/apex_os_polling_subscription_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED
#include "performance_test/communication_abstractions/rclcpp_waitset_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_FASTRTPS_ENABLED
#include "performance_test/communication_abstractions/fast_rtps_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED
#include "performance_test/communication_abstractions/connext_dds_micro_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_CONNEXTDDS_ENABLED
#include "performance_test/communication_abstractions/connext_dds_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_CYCLONEDDS_ENABLED
#include "performance_test/communication_abstractions/cyclonedds_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED
#include "performance_test/communication_abstractions/cyclonedds_cxx_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_ICEORYX_ENABLED
#include "performance_test/communication_abstractions/iceoryx_communicator.hpp"
#endif

#ifdef PERFORMANCE_TEST_OPENDDS_ENABLED
#include "performance_test/communication_abstractions/opendds_communicator.hpp"
#endif

namespace performance_test
{
std::shared_ptr<Publisher> CommunicatorFactory::get_publisher(
  const ExperimentConfiguration & ec)
{
  std::shared_ptr<Publisher> ptr;
  performance_test::for_each(
    messages::MessageTypeList(),
    [&ptr, &ec](const auto & msg_type) {
      using T = std::remove_cv_t<std::remove_reference_t<decltype(msg_type)>>;
      if (T::msg_name() == ec.msg_name()) {
        if (ptr) {
          throw std::runtime_error(
            "It seems that two msgs have the same name");
        }
#ifdef PERFORMANCE_TEST_RCLCPP_STE_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_SINGLE_THREADED_EXECUTOR) {
          ptr = std::make_shared<RclcppPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_STATIC_SINGLE_THREADED_EXECUTOR) {
          ptr = std::make_shared<RclcppPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_WAITSET) {
          ptr = std::make_shared<RclcppPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED
        if (ec.com_mean() == CommunicationMean::ApexOSPollingSubscription) {
          ptr = std::make_shared<RclcppPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_FASTRTPS_ENABLED
        if (ec.com_mean() == CommunicationMean::FASTRTPS) {
          ptr = std::make_shared<FastRTPSPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED
        if (ec.com_mean() == CommunicationMean::CONNEXTDDSMICRO) {
          ptr = std::make_shared<RTIMicroDDSPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CONNEXTDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::CONNEXTDDS) {
          ptr = std::make_shared<RTIDDSPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CYCLONEDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::CYCLONEDDS) {
          ptr = std::make_shared<CycloneDDSPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED
        if (ec.com_mean() == CommunicationMean::CYCLONEDDS_CXX) {
          ptr = std::make_shared<CycloneDDSCXXPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_ICEORYX_ENABLED
        if (ec.com_mean() == CommunicationMean::ICEORYX) {
          ptr = std::make_shared<IceoryxPublisher<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_OPENDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::OPENDDS) {
          ptr = std::make_shared<OpenDDSPublisher<T>>(ec);
        }
#endif
      }
    });
  if (!ptr) {
    throw std::runtime_error(
            "A topic with the requested name does not exist "
            "or communication mean not supported.");
  }
  return ptr;
}

std::shared_ptr<Subscriber> CommunicatorFactory::get_subscriber(
  const ExperimentConfiguration & ec)
{
  std::shared_ptr<Subscriber> ptr;
  performance_test::for_each(
    messages::MessageTypeList(),
    [&ptr, &ec](const auto & msg_type) {
      using T = std::remove_cv_t<std::remove_reference_t<decltype(msg_type)>>;
      if (T::msg_name() == ec.msg_name()) {
        if (ptr) {
          throw std::runtime_error(
            "It seems that two msgs have the same name");
        }
#ifdef PERFORMANCE_TEST_RCLCPP_STE_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_SINGLE_THREADED_EXECUTOR) {
          ptr = std::make_shared<RclcppSingleThreadedExecutorSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_STATIC_SINGLE_THREADED_EXECUTOR) {
          ptr = std::make_shared<RclcppStaticSingleThreadedExecutorSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED
        if (ec.com_mean() == CommunicationMean::RCLCPP_WAITSET) {
          ptr = std::make_shared<RclcppWaitsetSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED
        if (ec.com_mean() == CommunicationMean::ApexOSPollingSubscription) {
          ptr = std::make_shared<ApexOSPollingSubscriptionSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_FASTRTPS_ENABLED
        if (ec.com_mean() == CommunicationMean::FASTRTPS) {
          ptr = std::make_shared<FastRTPSSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED
        if (ec.com_mean() == CommunicationMean::CONNEXTDDSMICRO) {
          ptr = std::make_shared<RTIMicroDDSSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CONNEXTDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::CONNEXTDDS) {
          ptr = std::make_shared<RTIDDSSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CYCLONEDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::CYCLONEDDS) {
          ptr = std::make_shared<CycloneDDSSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED
        if (ec.com_mean() == CommunicationMean::CYCLONEDDS_CXX) {
          ptr = std::make_shared<CycloneDDSCXXSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_ICEORYX_ENABLED
        if (ec.com_mean() == CommunicationMean::ICEORYX) {
          ptr = std::make_shared<IceoryxSubscriber<T>>(ec);
        }
#endif
#ifdef PERFORMANCE_TEST_OPENDDS_ENABLED
        if (ec.com_mean() == CommunicationMean::OPENDDS) {
          ptr = std::make_shared<OpenDDSSubscriber<T>>(ec);
        }
#endif
      }
    });
  if (!ptr) {
    throw std::runtime_error(
            "A topic with the requested name does not exist "
            "or communication mean not supported.");
  }
  return ptr;
}

}  // namespace performance_test
