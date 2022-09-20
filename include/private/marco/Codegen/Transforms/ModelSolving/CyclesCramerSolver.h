
#include "marco/Codegen/Utils.h"
#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Modeling/AccessFunction.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

using namespace ::mlir::modelica;

namespace marco::codegen {

  /// SquareMatrix implements a square matrix that can be accessed using the
  /// parentheses operators.
  /// @size The number of rows and columns.
  /// @storage The array that contains the matrix values.
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

    /// Modify matrix elements with parentheses operator
    double& operator()(size_t row, size_t col)
    {
      return storage[row*size + col];
    }

    /// Access matrix elements with parentheses operator
    double operator()(size_t row, size_t col) const
    {
      return storage[row*size + col];
    }

    /// Print the matrix contents to stderr.
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

    /// The out matrix will contain the elements of the submatrix, minus the row
    /// and the column specified.
    /// \param out Output matrix.
    /// \param row Row to remove.
    /// \param col Column to remove.
    /// \return The out matrix.
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

    /// Substitute the column of the matrix indicated with the column number
    /// with the one specified by the col pointer, and return it in the out
    /// matrix given as input.
    /// \param out The matrix that will contain the result.
    /// \param colNumber The index of the matrix column we want to substitute.
    /// \param col The pointer to the vector that we want to substitute to the
    /// matrix column.
    /// \return The matrix obtained by substituting the given column in the
    /// current matrix at the specified column index
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

    /// Compute the determinant of the matrix
    /// \return The determinant of the matrix
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

  /// Get the matrix of the coefficients of the model and the constant vector
  /// \param matrix The matrix that will contain the model system coefficients.
  /// \param constantVector The vector that will contain the constant terms of
  /// the model equations.
  /// \param equations Cloned equations.
  /// \param builder The MLIR buillder.
  /// \return true if successful, false otherwise.
  bool getModelMatrixAndVector(
      SquareMatrix matrix,
      double* constantVector,
      Equations<MatchedEquation> equations,
      mlir::OpBuilder& builder)
  {

    /// Get the filtered variables from one of the equations
    assert(equations.size() > 0);
    Variables variables = equations[0]->getVariables();
    assert(variables.size() == equations.size());

    /// For each equation, get the coefficients of the variables and the
    /// constant term, and save them respectively in the system matrix and
    /// constant term vector.
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

    /// Solve the system of equations contained in the model using the Cramer
    /// method, if such model is linear in the variables.
    /// \param model The model containing the system of linear equations.
    /// \return true if successful, false otherwise.
    bool solve(Model<MatchedEquation>& model)
    {
      bool res = false;

      /// Clone the system equations so that we can operate on them without
      /// disrupting the rest of the compilation process.
      Equations<MatchedEquation> clones;
      for (const auto& equation : model.getEquations()) {
        auto clone = equation->clone();

        auto matchedClone = std::make_unique<MatchedEquation>(
            std::move(clone),
            equation->getIterationRanges(),
            equation->getWrite().getPath());

        clones.add(std::move(matchedClone));
      }

      /// Get the number of scalar equations in the system, which should be
      /// equal to the number of filtered variables.
      size_t numberOfScalarEquations = 0;
      for (const auto& equation : clones) {
        numberOfScalarEquations += equation->getIterationRanges().flatSize();
      }

      /// Create the matrix to contain the coefficients of the system's
      /// equations and the vector to contain the constant terms.
      double storage[numberOfScalarEquations*numberOfScalarEquations];
      auto matrix = SquareMatrix(storage, numberOfScalarEquations);
      double constantVector[numberOfScalarEquations];

      /// Populate system coefficient matrix and constant term vector.
      res = getModelMatrixAndVector(matrix, constantVector, clones, builder);

      if(res) {
        double solutionVector[numberOfScalarEquations];

        /// Create a temporary matrix to contain the matrices we are going to
        /// derive from the original one, by substituting the constant term
        /// vector to each of its columns.
        double tempStorage[numberOfScalarEquations*numberOfScalarEquations];
        auto temp = SquareMatrix(tempStorage, numberOfScalarEquations);

        /// Check that the system matrix is non singular
        double detA = matrix.det();
        if(detA == 0) return false;

        /// Compute the determinant of each one of the matrices obtained by
        /// substituting the constant term vector to each one of the matrix
        /// columns, and divide them by the determinant of the system matrix.
        for (size_t i = 0; i < numberOfScalarEquations; i++)
        {
          matrix.substituteColumn(temp, i, constantVector);
          double tempDet = temp.det();
          solutionVector[i] = tempDet/detA;
        }

        /// Set the results computed as the right side of the cloned equations,
        /// and the matched variables as the left side. Set the cloned equations
        /// as the model equations.
        //TODO: delete useless modelica instructions
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
          equation->setPath(EquationPath::LEFT);

        }

        model.setEquations(clones);
      }
      return res;
    }
  };
}