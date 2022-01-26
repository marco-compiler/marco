#include "marco/codegen/passes/model/Path.h"
#include <memory>

namespace marco::codegen
{
  /*
namespace marco::codegen
{
  class ExpressionPath::Step::Impl
  {
    public:
    virtual std::unique_ptr<ExpressionPath::Step::Impl> clone() const = 0;
  };
}
class RealStep : public ExpressionPath::Step::Impl
{
  public:
  std::unique_ptr<ExpressionPath::Step::Impl> clone() const override;
};
std::unique_ptr<ExpressionPath::Step::Impl> RealStep::clone() const
{
  return std::make_unique<RealStep>(*this);
}
class FakeStep : public ExpressionPath::Step::Impl
{
  public:
  std::unique_ptr<ExpressionPath::Step::Impl> clone() const override;
};
std::unique_ptr<ExpressionPath::Step::Impl> FakeStep::clone() const
{
  return std::make_unique<FakeStep>(*this);
}
ExpressionPath::Step::Step(std::unique_ptr<Impl> impl) : impl(std::move(impl))
{
}
ExpressionPath::Step::~Step() = default;
ExpressionPath::Step::Step(const ExpressionPath::Step& other) : impl(other.impl->clone())
{
}
 */

  ExpressionPath::Guard::Guard(ExpressionPath& path)
      : path(&path), size(path.size())
  {
  }

  ExpressionPath::Guard::~Guard()
  {
    if (path->size() > size)
    {
      size_t erase = path->size() - size;
      path->path.erase(path->path.begin(), std::next(path->path.begin(), erase));
    }
  }

  ExpressionPath::ExpressionPath(llvm::ArrayRef<size_t> path) : path(path.begin(), path.end())
  {
  }

  size_t ExpressionPath::operator[](size_t index) const
  {
    assert(index < path.size());
    return path[index];
  }

  size_t ExpressionPath::size() const
  {
    return path.size();
  }

  ExpressionPath::const_iterator ExpressionPath::begin() const
  {
    return path.begin();
  }

  ExpressionPath::const_iterator ExpressionPath::end() const
  {
    return path.end();
  }

  void ExpressionPath::append(size_t index)
  {
    path.push_back(index);
  }

  EquationPath::EquationPath(EquationSide equationSide, llvm::ArrayRef<size_t> path)
      : ExpressionPath(path), equationSide(equationSide)
  {
  }

  EquationPath::EquationSide EquationPath::getEquationSide() const
  {
    return equationSide;
  }
}
