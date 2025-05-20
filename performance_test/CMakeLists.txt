
# Copyright 2017-2024 Apex.AI, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.5)
project(performance_test)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
include(compile_options)

set(OPTIONAL_AMENT_DEPENDENCES)
set(OPTIONAL_LIBRARIES)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
find_package(ament_cmake REQUIRED)

find_package(Threads REQUIRED)

find_package(ament_cmake REQUIRED)

find_package(osrf_testing_tools_cpp QUIET)
if(${osrf_testing_tools_cpp_FOUND})
    list(APPEND OPTIONAL_AMENT_DEPENDENCES "osrf_testing_tools_cpp")
    list(APPEND OPTIONAL_LIBRARIES osrf_testing_tools_cpp::memory_tools)
    add_definitions(-DPERFORMANCE_TEST_MEMORYTOOLS_ENABLED)
endif()

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)

include_directories(include plugins ${osrf_testing_tools_cpp_INCLUDE_DIR})


# plugin selection
set(PERFORMANCE_TEST_PLUGIN "ROS2" CACHE STRING "Select a plugin")
set_property(
    CACHE PERFORMANCE_TEST_PLUGIN
    PROPERTY STRINGS
    "ROS2" "APEX_OS" "FASTDDS" "CONNEXTDDS" "CONNEXTDDSMICRO"
    "CYCLONEDDS" "CYCLONEDDS_CXX" "ICEORYX" "OPENDDS"
)
if(PERFORMANCE_TEST_PLUGIN MATCHES "ROS2")
  set(PERFORMANCE_TEST_RCLCPP_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "APEX_OS")
  set(PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "FASTDDS")
  set(PERFORMANCE_TEST_FASTRTPS_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "CONNEXTDDSMICRO")
  set(PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "CONNEXTDDS")
  set(PERFORMANCE_TEST_CONNEXTDDS_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "CYCLONEDDS_CXX")
  set(PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "CYCLONEDDS")
  set(PERFORMANCE_TEST_CYCLONEDDS_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "ICEORYX")
  set(PERFORMANCE_TEST_ICEORYX_ENABLED ON)
elseif(PERFORMANCE_TEST_PLUGIN MATCHES "OPENDDS")
  set(PERFORMANCE_TEST_OPENDDS_ENABLED ON)
else()
  message(
    FATAL_ERROR
    "Select a plugin with -DPERFORMANCE_TEST_PLUGIN=\"<plugin>\""
    "\nOptions are:"
    "\n- ROS2 (ROS 2 via rclcpp)"
    "\n- APEX_OS (Apex.OS via rclcpp)"
    "\n- FASTDDS (eProsima Fast DDS)"
    "\n- CONNEXTDDS (RTI Connext DDS Pro)"
    "\n- CONNEXTDDSMICRO (RTI Connext DDS Micro)"
    "\n- CYCLONEDDS (Eclipse Cyclone DDS)"
    "\n- CYCLONEDDS_CXX (Eclipse Cyclone DDS with C++ bindings)"
    "\n- ICEORYX (Eclipse iceoryx)"
    "\n- OPENDDS (Object Computing OpenDDS)"
  )
endif()


# ROS 2 rclcpp plugins
if(PERFORMANCE_TEST_RCLCPP_ENABLED)
  find_package(rclcpp REQUIRED)
  find_package(rmw REQUIRED)
  find_package(rosidl_default_generators REQUIRED)
  add_definitions(-DPERFORMANCE_TEST_RCLCPP_ENABLED)
  set(PERFORMANCE_TEST_RCLCPP_STE_ENABLED ON)
  set(PERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED ON)
  if($ENV{ROS_DISTRO} MATCHES "rolling")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    set(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED ON)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "jazzy")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    set(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED ON)
    add_definitions(-DPERFORMANCE_TEST_ROS2_JAZZY)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "iron")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    set(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED ON)
    add_definitions(-DPERFORMANCE_TEST_ROS2_IRON)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "humble")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    set(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED ON)
    add_definitions(-DPERFORMANCE_TEST_ROS2_HUMBLE)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "galactic")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    set(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED ON)
    add_definitions(-DPERFORMANCE_TEST_ROS2_GALACTIC)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "foxy")
    set(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED ON)
    add_definitions(-DPERFORMANCE_TEST_ROS2_FOXY)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "eloquent")
    set(PERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED OFF)
    add_definitions(-DPERFORMANCE_TEST_ROS2_ELOQUENT)
  endif()
  if($ENV{ROS_DISTRO} MATCHES "dashing")
    set(PERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED OFF)
    add_definitions(-DPERFORMANCE_TEST_ROS2_DASHING)
  endif()
  if(PERFORMANCE_TEST_RCLCPP_STE_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_STE_ENABLED)
  endif()
  if(PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_SSTE_ENABLED)
  endif()
  if(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED)
  endif()
  if(PERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED)
  endif()
endif()


# Apex.OS Waitset and Polling Subscription
if(PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED)
  find_library(APEXCPP_LIB apexcpp)
  if(NOT APEXCPP_LIB)
    message(WARNING "ApexOSPollingSubscription not found, disabling")
    set(PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED OFF)
  else()
    find_package(apexcpp REQUIRED)
    find_package(executor2 REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rmw REQUIRED)
    find_package(rosidl_default_generators REQUIRED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_ZERO_COPY_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED)
    add_definitions(-DPERFORMANCE_TEST_RCLCPP_LOANED_API_2_ENABLED)
  endif()
endif()

option(APEX_CERT "Set to true if building against Apex.OS Cert" OFF)
if(APEX_CERT)
  add_definitions(-DAPEX_CERT)
endif()


# FastDDS
if(PERFORMANCE_TEST_FASTRTPS_ENABLED)
  if($ENV{ROS_DISTRO} MATCHES "rolling|kilted")
    find_package(FastDDS QUIET REQUIRED)
    list(APPEND PLUGIN_LIBRARIES ${FastDDS_LIBRARIES})
  else()
    find_package(fastrtps_cmake_module QUIET REQUIRED)
    find_package(FastRTPS QUIET REQUIRED)
    list(APPEND PLUGIN_LIBRARIES ${FastRTPS_LIBRARIES})
  endif()
  add_definitions(-DPERFORMANCE_TEST_FASTRTPS_ENABLED)
endif()


# ConnextDDS
if(PERFORMANCE_TEST_CONNEXTDDS_ENABLED)
    # Append the path to CMAKE_MODULE_PATH in order to find the
    # FindConnextDDS.cmake CMake script
    set(CMAKE_MODULE_PATH
      ${CMAKE_MODULE_PATH}
      "$ENV{NDDSHOME}/resource/cmake"
    )

    find_package(RTIConnextDDS
      REQUIRED
      COMPONENTS
        core
    )
    add_definitions(-DPERFORMANCE_TEST_CONNEXTDDS_ENABLED)
endif()


# RTI Connext Micro
if(PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED)
  find_package(connext_micro_cmake_module REQUIRED)
  find_package(ConnextMicro MODULE REQUIRED)
  if(ConnextMicro_FOUND)
    add_definitions(-DPERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED)
    ament_export_definitions(${ConnextMicro_DEFINITIONS})
  endif()
endif()


# CycloneDDS
if(PERFORMANCE_TEST_CYCLONEDDS_ENABLED)
  find_package(CycloneDDS REQUIRED COMPONENTS idlc)
  add_definitions(-DPERFORMANCE_TEST_CYCLONEDDS_ENABLED)
  list(APPEND PLUGIN_LIBRARIES CycloneDDS::ddsc)
endif()


# cyclonedds-cxx
if(PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED)
  find_package(CycloneDDS-CXX REQUIRED)
  add_definitions(-DPERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED)
  list(APPEND PLUGIN_LIBRARIES cyclonedds_cxx_idl CycloneDDS-CXX::ddscxx)
endif()


# iceoryx
if(PERFORMANCE_TEST_ICEORYX_ENABLED)
  # iceoryx has no message generator, so performance_test uses ROS 2's
  find_package(rosidl_default_generators REQUIRED)
  find_package(iceoryx_posh CONFIG REQUIRED)
  add_definitions(-DPERFORMANCE_TEST_ICEORYX_ENABLED)
  list(APPEND PLUGIN_LIBRARIES iceoryx_posh::iceoryx_posh)
endif()


# OpenDDS
if(PERFORMANCE_TEST_OPENDDS_ENABLED)
  add_definitions(-DPERFORMANCE_TEST_OPENDDS_ENABLED)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()


# IDL file compilation
add_subdirectory(msg)
include_directories(${IDLGEN_INCLUDE_DIR})

if(PERFORMANCE_TEST_RCLCPP_ENABLED OR PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED)
  # rosidl_generate_interfaces must be invoked from this CMakeLists.txt file. Otherwise, the
  # generated messages are placed in a directory where ros1_bridge can not find them.
  rosidl_generate_interfaces(${PROJECT_NAME} ${ROSIDL_GEN_LIST})
endif()

# iceoryx has no message generator, so performance_test uses ROS 2's
if(PERFORMANCE_TEST_ICEORYX_ENABLED)
  # rosidl_generate_interfaces must be invoked from this CMakeLists.txt file. Otherwise, the
  # generated messages are placed in a directory where ros1_bridge can not find them.
  rosidl_generate_interfaces(${PROJECT_NAME} ${ROSIDL_GEN_LIST})
endif()


set(sources
    include/performance_test/cli/cli_parser.hpp
    include/performance_test/execution_tasks/publisher_task.hpp
    include/performance_test/execution_tasks/round_trip_relay_task.hpp
    include/performance_test/execution_tasks/subscriber_task.hpp
    include/performance_test/experiment_configuration/experiment_configuration.hpp
    include/performance_test/experiment_configuration/output_configuration.hpp
    include/performance_test/experiment_configuration/qos_abstraction.hpp
    include/performance_test/experiment_configuration/round_trip_mode.hpp
    include/performance_test/experiment_execution/data_entity_runner.hpp
    include/performance_test/experiment_execution/inter_thread_runner.hpp
    include/performance_test/experiment_execution/intra_thread_runner.hpp
    include/performance_test/experiment_execution/pub_sub_factory.hpp
    include/performance_test/experiment_execution/pub_sub_registry.hpp
    include/performance_test/experiment_execution/round_trip_main_runner.hpp
    include/performance_test/experiment_execution/round_trip_relay_runner.hpp
    include/performance_test/experiment_execution/runner.hpp
    include/performance_test/experiment_execution/runner_factory.hpp
    include/performance_test/experiment_execution/runner_registry.hpp
    include/performance_test/experiment_metrics/analysis_result.hpp
    include/performance_test/experiment_metrics/message_received_listener.hpp
    include/performance_test/experiment_metrics/publisher_stats.hpp
    include/performance_test/experiment_metrics/subscriber_stats.hpp
    include/performance_test/outputs/csv_output.hpp
    include/performance_test/outputs/json_output.hpp
    include/performance_test/outputs/output.hpp
    include/performance_test/outputs/output_factory.hpp
    include/performance_test/outputs/stdout_output.hpp
    include/performance_test/plugin/plugin_factory.hpp
    include/performance_test/plugin/plugin_singleton.hpp
    include/performance_test/plugin/plugin.hpp
    include/performance_test/plugin/publisher.hpp
    include/performance_test/plugin/subscriber.hpp
    include/performance_test/utilities/cpu_usage_tracker.hpp
    include/performance_test/utilities/exit_request_handler.hpp
    include/performance_test/utilities/external_info.hpp
    include/performance_test/utilities/for_each.hpp
    include/performance_test/utilities/memory_checker.hpp
    include/performance_test/utilities/message_initializer.hpp
    include/performance_test/utilities/msg_traits.hpp
    include/performance_test/utilities/perf_clock.hpp
    include/performance_test/utilities/qnx_res_usage.hpp
    include/performance_test/utilities/prevent_cpu_idle.hpp
    include/performance_test/utilities/rt_enabler.hpp
    include/performance_test/utilities/sample_statistics.hpp
    include/performance_test/utilities/spin_lock.hpp
    include/performance_test/utilities/version.hpp
    src/cli/cli_parser.cpp
    src/execution_tasks/publisher_task.cpp
    src/execution_tasks/round_trip_relay_task.cpp
    src/execution_tasks/subscriber_task.cpp
    src/experiment_configuration/experiment_configuration.cpp
    src/experiment_configuration/qos_abstraction.cpp
    src/experiment_configuration/round_trip_mode.cpp
    src/experiment_execution/data_entity_runner.cpp
    src/experiment_execution/inter_thread_runner.cpp
    src/experiment_execution/intra_thread_runner.cpp
    src/experiment_execution/pub_sub_factory.cpp
    src/experiment_execution/round_trip_main_runner.cpp
    src/experiment_execution/round_trip_relay_runner.cpp
    src/experiment_execution/runner.cpp
    src/experiment_execution/runner_factory.cpp
    src/experiment_metrics/analysis_result.cpp
    src/experiment_metrics/publisher_stats.cpp
    src/experiment_metrics/subscriber_stats.cpp
    src/main.cpp
    src/outputs/csv_output.cpp
    src/outputs/json_output.cpp
    src/outputs/output_factory.cpp
    src/outputs/stdout_output.cpp
    src/utilities/exit_request_handler.cpp
    src/utilities/external_info.cpp
    src/utilities/memory_checker.cpp
    src/utilities/prevent_cpu_idle.cpp
    src/utilities/rt_enabler.cpp
)

if(PERFORMANCE_TEST_RCLCPP_ENABLED)
  list(APPEND sources plugins/rclcpp_common/rclcpp_publisher.hpp)
  list(APPEND sources plugins/rclcpp_common/rclcpp_resource_manager.hpp)
  list(APPEND sources plugins/rclcpp_common/ros2_qos_adapter.hpp)
  list(APPEND sources plugins/ros2/plugin_factory.cpp)
  list(APPEND sources plugins/ros2/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_RCLCPP_STE_ENABLED OR PERFORMANCE_TEST_RCLCPP_SSTE_ENABLED)
  list(APPEND sources plugins/ros2/rclcpp_callback_communicator.hpp)
endif()

if(PERFORMANCE_TEST_RCLCPP_WAITSET_ENABLED)
  list(APPEND sources plugins/ros2/rclcpp_waitset_communicator.hpp)
endif()

if(PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED)
  list(APPEND sources plugins/apex_os/apex_os_communicator.hpp)
  list(APPEND sources plugins/apex_os/apex_os_entity_factory.hpp)
  list(APPEND sources plugins/apex_os/apex_os_polling_subscription_communicator.hpp)
  list(APPEND sources plugins/apex_os/apex_os_runner.hpp)
  list(APPEND sources plugins/apex_os/plugin_factory.cpp)
  list(APPEND sources plugins/apex_os/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_FASTRTPS_ENABLED)
  list(APPEND sources plugins/fastdds/fast_rtps_communicator.hpp)
  list(APPEND sources plugins/fastdds/plugin_factory.cpp)
  list(APPEND sources plugins/fastdds/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_CONNEXTDDSMICRO_ENABLED)
  list(APPEND sources plugins/connextdds_micro/connext_dds_micro_communicator.hpp)
  list(APPEND sources plugins/connextdds_micro/plugin_factory.cpp)
  list(APPEND sources plugins/connextdds_micro/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_CONNEXTDDS_ENABLED)
  list(APPEND sources plugins/connextdds/connext_dds_communicator.hpp)
  list(APPEND sources plugins/connextdds/plugin_factory.cpp)
  list(APPEND sources plugins/connextdds/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_CYCLONEDDS_ENABLED)
  list(APPEND sources plugins/cyclonedds/cyclonedds_communicator.hpp)
  list(APPEND sources plugins/cyclonedds/plugin_factory.cpp)
  list(APPEND sources plugins/cyclonedds/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_CYCLONEDDS_CXX_ENABLED)
  list(APPEND sources plugins/cyclonedds_cxx/cyclonedds_cxx_communicator.hpp)
  list(APPEND sources plugins/cyclonedds_cxx/plugin_factory.cpp)
  list(APPEND sources plugins/cyclonedds_cxx/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_ICEORYX_ENABLED)
  list(APPEND sources plugins/iceoryx/iceoryx_communicator.hpp)
  list(APPEND sources plugins/iceoryx/plugin_factory.cpp)
  list(APPEND sources plugins/iceoryx/plugin_impl.hpp)
endif()

if(PERFORMANCE_TEST_OPENDDS_ENABLED)
  list(APPEND sources plugins/opendds/opendds_communicator.hpp)
  list(APPEND sources plugins/opendds/plugin_factory.cpp)
  list(APPEND sources plugins/opendds/plugin_impl.hpp)
endif()

include(ExternalProject)

set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external)
externalproject_add(
  tclap
  GIT_REPOSITORY https://github.com/mirror/tclap.git
  GIT_TAG 1.4.0-rc1
  GIT_SHALLOW TRUE
  PREFIX ${EXTERNAL_INSTALL_LOCATION}
  SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}/tclap
  CMAKE_COMMAND ""
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
include_directories(SYSTEM ${EXTERNAL_INSTALL_LOCATION}/tclap/include)

externalproject_add(
  rapidjson
  GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
  GIT_TAG v1.1.0
  GIT_SHALLOW TRUE
  PREFIX ${EXTERNAL_INSTALL_LOCATION}
  SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}/rapidjson
  CMAKE_COMMAND ""
  CMAKE_ARGS
    -DRAPIDJSON_BUILD_TESTS=OFF
    -DRAPIDJSON_BUILD_DOC=OFF
    -DRAPIDJSON_BUILD_EXAMPLES=OFF
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
include_directories(SYSTEM ${EXTERNAL_INSTALL_LOCATION}/rapidjson/include)

externalproject_add(
  tabulate
  GIT_REPOSITORY https://github.com/p-ranav/tabulate.git
  GIT_TAG v1.4
  GIT_SHALLOW TRUE
  PREFIX ${EXTERNAL_INSTALL_LOCATION}
  SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}/tabulate
  PATCH_COMMAND git apply
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tabulate/diff.patch
  CMAKE_COMMAND ""
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
include_directories(SYSTEM ${EXTERNAL_INSTALL_LOCATION}/tabulate/include)

externalproject_add(
  sole
  GIT_REPOSITORY https://github.com/r-lyeh-archived/sole.git
  GIT_TAG 1.0.2
  GIT_SHALLOW TRUE
  PREFIX ${EXTERNAL_INSTALL_LOCATION}
  SOURCE_DIR ${EXTERNAL_INSTALL_LOCATION}/sole
  PATCH_COMMAND git apply
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sole/diff.patch
  CMAKE_COMMAND ""
  CONFIGURE_COMMAND ""
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
)
include_directories(SYSTEM ${EXTERNAL_INSTALL_LOCATION})

set(EXE_NAME perf_test)
add_executable(${EXE_NAME} ${sources})
add_dependencies(${EXE_NAME} tclap rapidjson tabulate sole)


set_compile_options(${EXE_NAME})


# Try to bake the githash into the perf_test EXE:
execute_process(
  COMMAND ./version_check.bash
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE PERF_TEST_VERSION
)
add_definitions(-DPERFORMANCE_TEST_VERSION="${PERF_TEST_VERSION}")

# iceoryx has no message generator, so performance_test uses ROS 2's
if(PERFORMANCE_TEST_RCLCPP_ENABLED OR PERFORMANCE_TEST_ICEORYX_ENABLED)
  if($ENV{ROS_DISTRO} MATCHES "rolling")
    rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(${EXE_NAME} ${cpp_typesupport_target})
  elseif($ENV{ROS_DISTRO} MATCHES "jazzy")
    rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(${EXE_NAME} ${cpp_typesupport_target})
  elseif($ENV{ROS_DISTRO} MATCHES "iron")
    rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(${EXE_NAME} ${cpp_typesupport_target})
  elseif($ENV{ROS_DISTRO} MATCHES "humble")
    rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
    target_link_libraries(${EXE_NAME} ${cpp_typesupport_target})
  elseif($ENV{ROS_DISTRO} MATCHES "galactic")
    rosidl_target_interfaces(${EXE_NAME} ${PROJECT_NAME} "rosidl_typesupport_cpp")
  elseif($ENV{ROS_DISTRO} MATCHES "foxy")
    rosidl_target_interfaces(${EXE_NAME} ${PROJECT_NAME} "rosidl_typesupport_cpp")
  elseif($ENV{ROS_DISTRO} MATCHES "eloquent")
    rosidl_target_interfaces(${EXE_NAME} ${PROJECT_NAME} "rosidl_typesupport_cpp")
  elseif($ENV{ROS_DISTRO} MATCHES "dashing")
    rosidl_target_interfaces(${EXE_NAME} ${PROJECT_NAME} "rosidl_typesupport_cpp")
  else()
    message(FATAL_ERROR "Unsupported ROS_DISTRO")
  endif()
endif()

if(PERFORMANCE_TEST_RCLCPP_ENABLED)
  ament_target_dependencies(${EXE_NAME} "rclcpp")
endif()

if(PERFORMANCE_TEST_APEX_OS_POLLING_SUBSCRIPTION_ENABLED)
  if(APEX_CERT)
    set(TYPESUPPORT_INTERFACE rosidl_typesupport_apex_middleware_cpp)
  else()
    set(TYPESUPPORT_INTERFACE rosidl_typesupport_cpp)
  endif()
  rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} ${TYPESUPPORT_INTERFACE})
  target_link_libraries(${EXE_NAME} ${cpp_typesupport_target})
  ament_target_dependencies(${EXE_NAME} "rclcpp" "executor2" "apexcpp")
endif()

ament_target_dependencies(${EXE_NAME} ${OPTIONAL_AMENT_DEPENDENCES})

target_link_libraries(
  ${EXE_NAME}
  ${OPTIONAL_LIBRARIES}
  ${PLUGIN_LIBRARIES}
  ${IDLGEN_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

if(BUILD_TESTING)
    find_package(ament_cmake_copyright REQUIRED)
    ament_copyright()

    find_package(ament_cmake_cppcheck REQUIRED)
    ament_cppcheck(
        ${sources})

    find_package(ament_cmake_cpplint REQUIRED)
    ament_cpplint(
        FILTERS "-build/include_order")

    find_package(ament_cmake_uncrustify REQUIRED)
    ament_uncrustify(
        ${sources})

    find_package(ament_cmake_lint_cmake REQUIRED)
    ament_lint_cmake()

    set(APEX_PERFORMANCE_TEST_GTEST apex_performance_test_gtest)

    find_package(ament_cmake_gtest REQUIRED)
    ament_add_gtest(${APEX_PERFORMANCE_TEST_GTEST}
        test/src/test_sample_statistics.cpp)

    target_link_libraries(${APEX_PERFORMANCE_TEST_GTEST})

    set_compile_options(${APEX_PERFORMANCE_TEST_GTEST})
endif()

install(TARGETS
    ${EXE_NAME}
    DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
    DESTINATION lib/${PROJECT_NAME})

install(FILES  mapping_rules.yaml
    DESTINATION share/${PROJECT_NAME})
ament_package()
