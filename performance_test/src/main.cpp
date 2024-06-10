// Copyright 2017-2024 Apex.AI, Inc.
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

#include <iostream>

#include "performance_test/cli/cli_parser.hpp"
#include "performance_test/experiment_execution/pub_sub_factory.hpp"
#include "performance_test/experiment_execution/runner_factory.hpp"
#include "performance_test/plugin/plugin_singleton.hpp"
#include "performance_test/utilities/exit_request_handler.hpp"
#include "performance_test/utilities/prevent_cpu_idle.hpp"
#include "performance_test/utilities/rt_enabler.hpp"

namespace pt = ::performance_test;

int main(int argc, char ** argv)
{
  pt::PluginSingleton::get()->register_pub_sub(pt::PubSubFactory::get());
  pt::PluginSingleton::get()->register_custom_runners(pt::RunnerFactory::get());

  pt::CLIParser parser(argc, argv);

  auto ec = parser.experiment_configuration;

  // TODO (Sumanth.Nirmal) disable rt settings
  // if (ec.rt_config.is_rt_init_required()) {
  //   pt::pre_proc_rt_init(ec.rt_config.cpus, ec.rt_config.prio);
  // }

  if (ec.prevent_cpu_idle) {
    pt::prevent_cpu_idle();
  }

  pt::PluginSingleton::get()->global_setup(ec);

  pt::ExitRequestHandler::get().setup();

  {
    auto r = pt::RunnerFactory::get().create_runner(ec);

    // TODO (Sumanth.Nirmal) disable rt settings
    // if (ec.rt_config.is_rt_init_required()) {
    //   pt::post_proc_rt_init();
    // }

    r->run();
  }

  pt::PluginSingleton::get()->global_teardown(ec);
}
