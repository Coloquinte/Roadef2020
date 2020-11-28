
#pragma once

#include "problem.hpp"

/*
 * Heuristic measures to make decisions about a problem
 */

std::vector<double> measureSpanMeanRisk(const Problem &pb);
std::vector<double> measureSpanMinRisk(const Problem &pb);
std::vector<double> measureSpanMedianRisk(const Problem &pb);
std::vector<double> measureSpanQuantileRisk(const Problem &pb);
std::vector<double> measureSpanMaxRisk(const Problem &pb);
std::vector<double> measureAverageDemand(const Problem &pb);
std::vector<double> measureAverageDuration(const Problem &pb);
std::vector<double> measureValidTimestepRatio(const Problem &pb);
std::vector<std::vector<int> > validTimesteps(const Problem &pb);

std::vector<double> measureSpanMeanRisk(const Problem &pb, double objectiveBound);
std::vector<double> measureSpanMinRisk(const Problem &pb, double objectiveBound);
std::vector<double> measureSpanMedianRisk(const Problem &pb, double objectiveBound);
std::vector<double> measureSpanQuantileRisk(const Problem &pb, double objectiveBound);
std::vector<double> measureSpanMaxRisk(const Problem &pb, double objectiveBound);
std::vector<double> measureAverageDemand(const Problem &pb, double objectiveBound);
std::vector<double> measureAverageDuration(const Problem &pb, double objectiveBound);
std::vector<double> measureValidTimestepRatio(const Problem &pb, double objectiveBound);
std::vector<std::vector<int> > validTimesteps(const Problem &pb, double objectiveBound);

