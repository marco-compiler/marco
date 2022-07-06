#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ALGORITHM_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ALGORITHM_H

#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/ModelSolving/Access.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class Algorithm
  {
    public:
      Algorithm(mlir::modelica::AlgorithmOp op, Variables variables);

      Algorithm(const Algorithm& other);

      ~Algorithm();

      Algorithm& operator=(const Algorithm& other);
      Algorithm& operator=(Algorithm&& other);

      friend void swap(Algorithm& first, Algorithm& second);

      std::unique_ptr<Algorithm> clone() const;

      void dumpIR() const;

      void dumpIR(llvm::raw_ostream& os) const;

      mlir::modelica::AlgorithmOp getOperation() const;

      Variables getVariables() const;

      void setVariables(Variables variables);

      std::vector<Access> getAccesses() const;

      std::vector<Access> getWrites() const;

      std::vector<Access> getReads() const;

    private:
      bool isVariable(mlir::Value value) const;

      bool isReferenceAccess(mlir::Value value) const;

    private:
      mlir::Operation* operation;
      Variables variables;
  };

  class Algorithms
  {
    private:
      using Container = std::vector<Algorithm>;

    public:
      using iterator = typename Container::iterator;
      using const_iterator = typename Container::const_iterator;

      iterator begin()
      {
        return algorithms.begin();
      }

      const_iterator begin() const
      {
        return algorithms.begin();
      }

      iterator end()
      {
        return algorithms.end();
      }

      const_iterator end() const
      {
        return algorithms.end();
      }

    private:
      Container algorithms;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_ALGORITHM_H
