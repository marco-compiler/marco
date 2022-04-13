#ifndef MARCO_TRANSFORMS_MODELSOLVING_EXTERNALSOLVERS_IDAOPTIONS_H
#define MARCO_TRANSFORMS_MODELSOLVING_EXTERNALSOLVERS_IDAOPTIONS_H

namespace marco::codegen
{
  struct IDAOptions
  {
    double relativeTolerance = 1e-06;
    double absoluteTolerance = 1e-06;
    bool equidistantTimeGrid = false;
  };
}

#endif // MARCO_TRANSFORMS_MODELSOLVING_EXTERNALSOLVERS_IDAOPTIONS_H
