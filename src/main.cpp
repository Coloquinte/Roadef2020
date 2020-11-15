// Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

#include "problem.hpp"
#include "bs_optimizer.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

po::options_description getOptions() {
  po::options_description desc("Options");

  desc.add_options()("instance,p", po::value<string>()->required(),
                     "Input file name (.json)");

  desc.add_options()("solution,o", po::value<string>()->required(),
                     "Solution file name (.txt)");

  desc.add_options()("seed,s", po::value<size_t>()->default_value(0),
                     "Random seed");

  desc.add_options()("time-limit,t", po::value<double>()->default_value(numeric_limits<double>::infinity()),
                     "Time limit");

  desc.add_options()("name", "Print the team's name (J3)");
  desc.add_options()("help,h", "Print this help");

  desc.add_options()("verbosity,v", po::value<int>()->default_value(2),
                     "Verbosity level\n    1 -> basic stats\n    2 -> new solutions\n    3 -> search process");

  return desc;
}

po::variables_map parseArguments(int argc, char **argv) {
  cout << fixed << setprecision(3);
  cerr << fixed << setprecision(3);

  po::options_description options = getOptions();

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv)
        .options(options)
        .style(po::command_line_style::unix_style
             | po::command_line_style::allow_long_disguise)
        .run(), vm);

    if (vm.count("help")) {
      cout << "Roadef Optimizer J3 (Gabriel Gouvine)" << endl;
      cout << options << endl;
      exit(0);
    }
    if (vm.count("name")) {
      cout << "J3" << endl;
      exit(0);
    }

    po::notify(vm);
  } catch (po::error &e) {
    cerr << "Error parsing command line arguments: ";
    cerr << e.what() << endl << endl;
    cout << options << endl;
    exit(1);
  }

  return vm;
}

RoadefParams readParams(const po::variables_map &vm) {
  return RoadefParams {
    .instance = vm["instance"].as<string>(),
    .solution = vm["solution"].as<string>(),
    .verbosity = vm["verbosity"].as<int>(),
    .seed = vm["seed"].as<size_t>(),
    .time_limit = vm["time-limit"].as<double>(),
  };
}

int main(int argc, char **argv) {
  po::variables_map vm = parseArguments(argc, argv);

  RoadefParams params = readParams(vm);
  Problem pb = Problem::readFile(params.instance);

  if (params.verbosity >= 1) {
    cout << "Problem with "
         << pb.nbInterventions() << " interventions "
         << pb.nbResources() << " resources "
         << pb.nbTimesteps() << " timesteps "
         << endl;
    cout << "Random seed is " << params.seed << endl;
    if (isfinite(params.time_limit)) {
        cout << "Time limit is " << params.time_limit << endl;
    }
    else {
        cout << "No time limit" << endl;
    }
  }

  BsOptimizer opti(pb, params);
  opti.run();

  if (params.verbosity >= 1) {
    cout << "Solution with "
         << pb.exclusionValue() << " exclusions, "
         << pb.resourceValue() << " overflow, "
         << pb.riskValue() << " risk "
         << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")"
         << endl;
  }

  return 0;
}

