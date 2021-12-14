#ifndef MARCO_CODEGEN_PATH_H
#define MARCO_CODEGEN_PATH_H

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace marco::codegen
{
  class ExpressionPath
  {
    public:
      /*
    class Step
    {
      public:
      class Impl;

      ~Step();

      Step(const Step& other);

      static Step real(size_t index);
      static Step fake(size_t index);

      size_t getIndex() const;

      private:
      Step(std::unique_ptr<Impl> impl);

      std::unique_ptr<Impl> impl;
    };
       */

    class Guard
    {
      public:
      Guard(ExpressionPath& path);

      ~Guard();

      private:
      ExpressionPath* path;
      size_t size;
    };

    ExpressionPath(llvm::ArrayRef<size_t> path);

    size_t operator[](size_t index) const;
    size_t size() const;

    void append(size_t index);

    private:
    llvm::SmallVector<size_t, 3> path;
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
