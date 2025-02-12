#include "marco/VariableFilter/Range.h"
#include <cassert>

using namespace ::marco;
using namespace ::marco::vf;

namespace marco::vf {
Range::Range(long lowerBound, long upperBound)
    : lowerBound(lowerBound), upperBound(upperBound) {}

bool Range::hasLowerBound() const { return lowerBound != kUnbounded; }

long Range::getLowerBound() const {
  assert(hasLowerBound());
  return lowerBound;
}

bool Range::hasUpperBound() const { return upperBound != kUnbounded; }

long Range::getUpperBound() const {
  assert(hasUpperBound());
  return upperBound;
}
} // namespace marco::vf
