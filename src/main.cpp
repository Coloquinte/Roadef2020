// Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

#include "problem.hpp"
#include "ls_optimizer.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

po::options_description getOptions() {
  po::options_description desc("Options");

  desc.add_options()("instance", po::value<string>()->required(),
                     "Input file name (.json)");

  desc.add_options()("solution", po::value<string>()->required(),
                     "Solution file name (.txt)");

  desc.add_options()("verbosity,v", po::value<int>()->default_value(1),
                     "Verbosity level");

  desc.add_options()("seed,s", po::value<size_t>()->default_value(0),
                     "Random seed");

  desc.add_options()("help,h", "Print this help");

  return desc;
}

po::variables_map parseArguments(int argc, char **argv) {
  cout << fixed << setprecision(1);
  cerr << fixed << setprecision(1);

  po::options_description options = getOptions();
  po::positional_options_description pos;
  pos.add("instance", 1);
  pos.add("solution", 1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(options).positional(pos).run(), vm);
    po::notify(vm);
  } catch (po::error &e) {
    cerr << "Error parsing command line arguments: ";
    cerr << e.what() << endl << endl;
    cout << options << endl;
    exit(1);
  }

  if (vm.count("help")) {
    cout << "Roadef Optimizer J3 (Gabriel Gouvine)" << endl;
    exit(0);
  }

  return vm;
}

RoadefParams readParams(const po::variables_map &vm) {
  return RoadefParams {
    .instance = vm["instance"].as<string>(),
    .solution = vm["solution"].as<string>(),
    .verbosity = vm["verbosity"].as<int>(),
    .seed = vm["seed"].as<size_t>(),
  };
}

int main(int argc, char **argv) {
  po::variables_map vm = parseArguments(argc, argv);

  RoadefParams params = readParams(vm);
  Problem pb = Problem::readFile(params.instance);

  cout << "Problem with "
       << pb.nbInterventions() << " interventions "
       << pb.nbResources() << " resources "
       << pb.nbTimesteps() << " timesteps "
       << endl;

  LsOptimizer opti(pb, params);
  opti.run();

  return 0;
}

