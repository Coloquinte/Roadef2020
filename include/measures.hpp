
#pragma once

#include "problem.hpp"

/*
 * Heuristic measures to make decisions about a problem
 *
 */

std::vector<double> measureSpanMeanRisk(const Problem &pb);
std::vector<double> measureAverageDemand(const Problem &pb);
std::vector<double> measureAverageDuration(const Problem &pb);
std::vector<double> measureValidTimestepRatio(const Problem &pb);
std::vector<std::vector<int> > validTimesteps(const Problem &pb);

