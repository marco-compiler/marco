
#include "marco/Codegen/Utils.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/AccessFunction.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

using namespace ::mlir::modelica;

namespace marco::codegen {

  class SquareMatrix
  {
  private:
    size_t size;
    double* storage;

  public:

    SquareMatrix(double* storage, size_t size) : size(size), storage(storage)
    {
    }

    size_t getSize()
    {
      return size;
    }

    double& operator()(size_t row, size_t col)
    {
      return storage[row*size + col];
    }

    double operator()(size_t row, size_t col) const
    {
      return storage[row*size + col];
    }

    void dump()
    {
      std::cerr << "Matrix: \n";
      for (size_t i = 0; i < size; i++)
      {
        for (size_t j = 0; j < size; j++)
        {
          std::cerr << (*this)(i,j) << " ";
        }
        std::cerr << "\n";
      }
      std::cerr << "\n" << std::flush;
    }

    SquareMatrix subMatrix(SquareMatrix out, size_t row, size_t col)
    {
      for (size_t i = 0; i < size; i++)
      {
        if(i != row)
        {
          for (size_t j = 0; j < size; j++)
          {
            if(j != col)
            {
              out(i < row ? i : i - 1, j < col ? j : j - 1) = (*this)(i, j);
            }
          }
        }
      }

      return out;
    }

    SquareMatrix substituteColumn(SquareMatrix out, size_t colNumber, double* col)
    {
      assert(colNumber < size && out.getSize() == size);
      for (size_t i = 0; i < size; i++)
      {
        for (size_t j = 0; j < size; j++)
        {
          if(j != colNumber)
            out(i,j) = (*this)(i,j);
          else
            out(i,j) = col[i];
        }
      }

      return out;
    }

    double det()
    {
      double determinant = 0;

      SquareMatrix matrix = (*this);

      if(size == 1)
        determinant = matrix(0,0);
      else if(size == 2) {
        determinant = matrix(0,0)*matrix(1,1) - matrix(0,1)*matrix(1,0);
      }
      else if (size > 2) {
        int sign = 1;
        double storage[(size - 1)*(size -1)];
        SquareMatrix out = SquareMatrix(storage, size-1);
        for (volatile int i = 0; i < size; i++) {
          determinant += sign * matrix(0,i) * subMatrix(out, 0, i).det();
          sign = -sign;
        }

      }

      return determinant;
    }
  };

  /// Get the matrix model and constant vector
  /// @equations Cloned and explicitated equations
  bool getModelMatrixAndVector(SquareMatrix matrix,
                               double* constantVector,
                               Equations<MatchedEquation> equations,
                               mlir::OpBuilder& builder)
  {

    assert(equations.size() > 0);
    Variables variables = equations[0]->getVariables();

    assert(variables.size() == equations.size());

    bool check = true;


    for(size_t i = 0; i < equations.size(); ++i) {

      auto equation = equations[i].get();
      auto vector = std::vector<double>();
      double constantTerm;

      auto res = equation->getCoefficients(builder, vector, constantTerm);

      if(mlir::failed(res)) {
        return false;
      }

      assert(vector.size() == variables.size());

      for(size_t j = 0; j < variables.size(); ++j) {
        matrix(i, j) = vector[j];
      }

      constantVector[i] = constantTerm;

    }

    return true;
  }

  class CramerSolver
  {
  private:
    mlir::OpBuilder& builder;
  public:
    CramerSolver(mlir::OpBuilder& builder) : builder(builder)
    {
    }

    bool solve(Model<MatchedEquation>& model)
    {
      bool res = false;

      size_t numberOfScalarEquations = 0;

      Equations<MatchedEquation> clones;

      for (const auto& equation : model.getEquations()) {
        auto clone = equation->clone();

        auto matchedClone = std::make_unique<MatchedEquation>(
            std::move(clone),
            equation->getIterationRanges(),
            equation->getWrite().getPath());

        clones.add(std::move(matchedClone));
      }

      for (const auto& equation : clones) {
        numberOfScalarEquations += equation->getIterationRanges().flatSize();
      }

      double storage[numberOfScalarEquations*numberOfScalarEquations];
      auto matrix = SquareMatrix(storage, numberOfScalarEquations);
      double constantVector[numberOfScalarEquations];

      res = getModelMatrixAndVector(matrix, constantVector, clones, builder);

      if(res) {
        double solutionVector[numberOfScalarEquations];

        double tempStorage[numberOfScalarEquations*numberOfScalarEquations];
        auto temp = SquareMatrix(tempStorage, numberOfScalarEquations);
        double detA = matrix.det();

        if(detA == 0) return false;

        for (size_t i = 0; i < numberOfScalarEquations; i++)
        {
          matrix.substituteColumn(temp, i, constantVector);
          double tempDet = temp.det();
          solutionVector[i] = tempDet/detA;
        }

        for(const auto& equation : clones) {
          auto variable = equation->getWrite().getVariable();
          auto argument = variable->getValue().cast<mlir::BlockArgument>();

          auto terminator =
              mlir::cast<EquationSidesOp>(equation->getOperation().bodyBlock()->getTerminator());

          builder.setInsertionPoint(terminator);

          auto lhs = equation->getValueAtPath(equation->getWrite().getPath());

          auto rhs = builder.create<ConstantOp>(
              model.getOperation().getLoc(),
              RealAttr::get(builder.getContext(),
                            solutionVector[argument.getArgNumber()]));

          equation->replaceSides(builder, lhs, rhs.getResult());
          equation->setDefaultMatchedPath();

        }

        model.setEquations(clones);
      }
      return res;
    }
  };
}