#ifndef MARCO_CODEGEN_PATH_H
#define MARCO_CODEGEN_PATH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace marco::codegen
{
  class ExpressionPath
  {
    public:
      class Guard
      {
        public:
          Guard(ExpressionPath& path);

          ~Guard();

        private:
          ExpressionPath* path;
          size_t size;
      };

    private:
      using Container = llvm::SmallVector<size_t, 3>;

    public:
      using const_iterator = Container::const_iterator;

      ExpressionPath(llvm::ArrayRef<size_t> path);

      size_t operator[](size_t index) const;
      size_t size() const;

      const_iterator begin() const;
      const_iterator end() const;

      void append(size_t index);

    private:
      Container path;
  };

  class EquationPath : public ExpressionPath
  {
    public:
      enum EquationSide
      {
        LEFT,
        RIGHT
      };

      using Guard = ExpressionPath::Guard;

      EquationPath(EquationSide equationSide, llvm::ArrayRef<size_t> path = llvm::None);

      EquationSide getEquationSide() const;

    private:
      EquationSide equationSide;
  };
}

#endif // MARCO_CODEGEN_PATH_H
