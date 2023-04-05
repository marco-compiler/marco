#include "marco/Codegen/Lowering/Results.h"

using namespace ::marco::codegen;

namespace marco::codegen::lowering
{
  Result::Result(Reference reference)
      : reference(std::move(reference))
  {
  }

  mlir::Location Result::getLoc() const
  {
    return reference.getLoc();
  }

  mlir::Value Result::getReference() const
  {
    return reference.getReference();
  }

  mlir::Value Result::get(mlir::Location loc) const
  {
    return reference.get(loc);
  }

  void Result::set(mlir::Location loc, mlir::Value value)
  {
    reference.set(loc, value);
  }

  Results::Results() = default;

  Results::Results(Reference value)
  {
    values.emplace_back(std::move(value));
  }

  Result& Results::operator[](size_t index)
  {
    assert(index < values.size());
    return values[index];
  }

  const Result& Results::operator[](size_t index) const
  {
    assert(index < values.size());
    return values[index];
  }

  void Results::append(Reference value)
  {
    values.emplace_back(std::move(value));
  }

  void Results::append(Result value)
  {
    values.emplace_back(std::move(value));
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
