// Copyright (C) 2019 Gabriel Gouvine - All Rights Reserved

#include "problem.hpp"
#include "bs_optimizer.hpp"
#include "bs_analyzer.hpp"

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

  desc.add_options()("output,o", po::value<string>()->required(),
                     "Output file name (.txt)");

  desc.add_options()("seed,s", po::value<size_t>()->default_value(0),
                     "Random seed");

  desc.add_options()("time-limit,t", po::value<double>(),
                     "Time limit");

  desc.add_options()("verbosity,v", po::value<int>()->default_value(2),
                     "Verbosity level\n    1 -> basic stats\n    2 -> new solutions\n    3 -> parameters\n    4 -> search process");

  desc.add_options()("name", "Print the team's name (J3)");
  desc.add_options()("help,h", "Print this help");

  desc.add_options()("beam-width", po::value<double>()->default_value(10.0),
                     "Beam width during search");

  desc.add_options()("backtrack-depth", po::value<double>()->default_value(0.5),
                     "Depth used to backtrack during search");

  desc.add_options()("restart-depth", po::value<double>()->default_value(10.0),
                     "Depth used to backtrack during restarts");

  desc.add_options()("warm-start", po::value<vector<string> >()->multitoken(),
                     "Warm-start from the solution file");

  desc.add_options()("analyze", "Analyze the given solution");

  return desc;
}

po::variables_map parseArguments(int argc, char **argv) {
  po::options_description options = getOptions();

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv)
        .options(options)
        .style((po::command_line_style::unix_style | po::command_line_style::allow_long_disguise)
              & ~po::command_line_style::allow_guessing & ~po::command_line_style::allow_sticky)
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
    double timeLimit = vm.count("time-limit") ? vm["time-limit"].as<double>() : 1.0e8;
    chrono::steady_clock::time_point startTime = chrono::steady_clock::now();
    chrono::steady_clock::time_point endTime = startTime + chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(0.99 * timeLimit));
    return RoadefParams {
      .instance = vm["instance"].as<string>(),
      .solution = vm["output"].as<string>(),
      .verbosity = vm["verbosity"].as<int>(),
      .seed = vm["seed"].as<size_t>(),
      .timeLimit = timeLimit,
      .startTime = startTime,
      .endTime = endTime,
      .beamWidth = vm["beam-width"].as<double>(),
      .backtrackDepth = vm["backtrack-depth"].as<double>(),
      .restartDepth = vm["restart-depth"].as<double>(),
    };
}

int main(int argc, char **argv) {
  po::variables_map vm = parseArguments(argc, argv);

  RoadefParams params = readParams(vm);
  Problem pb = Problem::readFile(params.instance);
  if (vm.count("analyze")) {
      pb.readSolutionFile(params.solution);
      BsAnalyzer(pb, params).run();
      exit(0);
  }
  if (vm.count("warm-start")) {
    vector<string> warmStarts = vm["warm-start"].as<vector<string> >();
    Problem::Objective bestObj;
    vector<int> bestStartTimes;
    for (string f : warmStarts) {
        pb.readSolutionFile(f);
        if (pb.validSolution() && pb.objective().betterThan(bestObj)) {
            bestObj = pb.objective();
            bestStartTimes = pb.startTimes();
        }
    }
    if (bestStartTimes.size() != 0) {
        pb.reset(bestStartTimes);
    }
  }

  if (params.verbosity >= 1) {
    if (params.verbosity >= 3) {
      cout << "Random seed set to " << params.seed << ". ";
      if (params.timeLimit < 1e8) {
          cout << "Time limit set to " << fixed << setprecision(1) << params.timeLimit << "s. ";
      }
      else {
          cout << "Time limit not set. ";
      }
      chrono::duration<double> elapsed = chrono::steady_clock::now() - params.startTime;
      cout << "Parsing took " << fixed << setprecision(1) << elapsed.count() << "s. ";
      cout << endl;
    }
    cout << "BEAM: problem with "
         << pb.nbInterventions() << " interventions, "
         << pb.nbResources() << " resources, "
         << pb.nbTimesteps() << " timesteps "
         << endl;
  }

  BsOptimizer opti(pb, params);
  opti.run();

  if (params.verbosity >= 1) {
    cout << "BEAM: final solution with "
         << "risk " << fixed << setprecision(5) << pb.riskValue() << " "
         << fixed << setprecision(2) << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << "), "
         << "exclusions " << pb.exclusionValue() << ", "
         << "overflow " << fixed << setprecision(2) << pb.resourceValue()
         << endl;
  }

  return 0;
}

