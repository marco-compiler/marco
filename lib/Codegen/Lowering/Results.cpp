#include "marco/Codegen/Lowering/Results.h"

using namespace ::marco::codegen;

namespace marco::codegen::lowering
{
  Results::Results() = default;

  Results::Results(Reference value)
  {
    values.push_back(std::move(value));
  }

  Reference& Results::operator[](size_t index)
  {
    assert(index < values.size());
    return values[index];
  }

  const Reference& Results::operator[](size_t index) const
  {
    assert(index < values.size());
    return values[index];
  }

  void Results::append(Reference value)
  {
    values.push_back(std::move(value));
  }

  size_t Results::size() const
  {
    return values.size();
  }

  Results::iterator Results::begin()
  {
    return values.begin();
  }

  Results::const_iterator Results::begin() const
  {
    return values.begin();
  }

  Results::iterator Results::end()
  {
    return values.end();
  }

  Results::const_iterator Results::end() const
  {
    return values.end();
  }
}
