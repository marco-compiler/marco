#include "marco/Codegen/Transforms/ModelSolving/Path.h"
#include <memory>

namespace marco::codegen
{
  ExpressionPath::Guard::Guard(ExpressionPath& path)
      : path(&path), size(path.size())
  {
  }

  ExpressionPath::Guard::~Guard()
  {
    if (path->size() > size) {
      path->path.erase(std::next(path->path.begin(), size), path->path.end());
    }
  }

  ExpressionPath::ExpressionPath(llvm::ArrayRef<size_t> path) : path(path.begin(), path.end())
  {
  }

  bool ExpressionPath::operator==(const ExpressionPath& other) const
  {
    return path == other.path;
  }

  bool ExpressionPath::operator!=(const ExpressionPath& other) const
  {
    return path != other.path;
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

  bool EquationPath::operator==(const EquationPath& other) const
  {
    if (static_cast<const ExpressionPath&>(*this) != static_cast<const ExpressionPath&>(other)) {
      return false;
    }

    return equationSide == other.equationSide;
  }

  bool EquationPath::operator!=(const EquationPath& other) const
  {
    if (static_cast<const ExpressionPath&>(*this) != static_cast<const ExpressionPath&>(other)) {
      return true;
    }

    return equationSide != other.equationSide;
  }

  void EquationPath::dump() const
  {
    dump(std::clog);
  }

  void EquationPath::dump(std::ostream& os) const
  {
    os << "Equation path: [";

    if (equationSide == LEFT) {
      os << "left";
    } else {
      os << "right";
    }

    for (const auto& index : *this) {
      os << ", " << index;
    }

    os << "]\n";
  }

  EquationPath::EquationSide EquationPath::getEquationSide() const
  {
    return equationSide;
  }
}
