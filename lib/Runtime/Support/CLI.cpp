#include "marco/Runtime/Support/CLI.h"
#include "marco/Runtime/Support/Options.h"
#include <iostream>

namespace marco::runtime::support
{
  std::string ApproximationOptions::getTitle() const
  {
    return "Approximation";
  }

  void ApproximationOptions::printCommandLineOptions(
      std::ostream& os) const
  {
    os << "  --enable-sin-interpolation           Enable the small-arcs approximation for sine computation." << std::endl;
    os << "  --sin-interpolation-points=<value>   Set the number of points within a quadrant to be exactly pre-computed for the sine approximation." << std::endl;
    os << "  --enable-cos-interpolation           Enable the small-arcs approximation for cosine computation." << std::endl;
    os << "  --cos-interpolation-points=<value>   Set the number of points within a quadrant to be exactly pre-computed for the cosine approximation." << std::endl;
  }

  void ApproximationOptions::parseCommandLineOptions(
      const argh::parser& options) const
  {
    supportOptions().useSinInterpolation = options["enable-sin-interpolation"];
    options("sin-interpolation-points") >> supportOptions().sinInterpolationPoints;
    supportOptions().useCosInterpolation = options["enable-cos-interpolation"];
    options("cos-interpolation-points") >> supportOptions().cosInterpolationPoints;
  }

  std::unique_ptr<cli::Category> getCLIApproximationOptions()
  {
    return std::make_unique<ApproximationOptions>();
  }
}
