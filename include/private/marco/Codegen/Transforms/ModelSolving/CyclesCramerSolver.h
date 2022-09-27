
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
    std::vector<mlir::Attribute>& storage;

  public:

    SquareMatrix(std::vector<mlir::Attribute>& storage, size_t size) : size(size), storage(storage)
    {
    }

    size_t getSize()
    {
      return size;
    }

    /// Modify matrix elements with parentheses operator
    mlir::Attribute& operator()(size_t row, size_t col)
    {
      return storage[row*size + col];
    }

    /// Access matrix elements with parentheses operator
    mlir::Attribute operator()(size_t row, size_t col) const
    {
      return storage[row*size + col];
    }

    /// Print the matrix contents to stderr.
    void dump()
    {
      std::cerr << "Matrix size: " << getSize() << "\n";
      for (size_t i = 0; i < size; i++)
      {
        for (size_t j = 0; j < size; j++)
        {
          (*this)(i,j).dump();
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
    SquareMatrix substituteColumn(SquareMatrix out, size_t colNumber, std::vector<mlir::Attribute>& col)
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
    mlir::Attribute det(mlir::OpBuilder& builder)
    {
      //TODO How do I decide the type?
      mlir::OpBuilder::InsertionGuard guard(builder);

      auto type = RealAttr::get(builder.getContext(), 0).getType();

      SquareMatrix matrix = (*this);
      mlir::Value determinant;
      mlir::Location loc = builder.getUnknownLoc();


      if(size == 1)
        determinant = builder.create<ConstantOp>(
            loc,
            matrix(0,0));

      else if(size == 2) {
        auto a = builder.create<ConstantOp>(
            loc, matrix(0,0));
        auto b = builder.create<ConstantOp>(
            loc, matrix(0,1));
        auto c = builder.create<ConstantOp>(
            loc, matrix(1,0));
        auto d = builder.create<ConstantOp>(
            loc, matrix(1,1));

        auto ad = builder.createOrFold<MulOp>(
            loc, type, a, d);
        auto bc = builder.createOrFold<MulOp>(
            loc, type, b, c);

        determinant = builder.createOrFold<SubOp>(
            loc, type, ad, bc);
      }
      else if (size > 2) {
        int sign = 1;
        int subMatrixSize = size - 1;

        auto zeroAttr = RealAttr::get(builder.getContext(), 0);
        auto subMatrixStorage = std::vector<mlir::Attribute>(subMatrixSize*subMatrixSize, zeroAttr);
        SquareMatrix out = SquareMatrix(subMatrixStorage, subMatrixSize);

        determinant = builder.create<ConstantOp>(
            loc, zeroAttr);

        for (volatile int i = 0; i < size; i++) {
          auto scalarDet = builder.create<ConstantOp>(
              loc, matrix(0,i));

          auto matrixDet = builder.create<ConstantOp>(
              loc, subMatrix(out, 0, i).det(builder));

          mlir::Value product = builder.createOrFold<MulOp>(
              loc, type, scalarDet, matrixDet);

          if(sign == -1)
            product = builder.createOrFold<NegateOp>(
                loc, type, product);

          determinant = builder.createOrFold<AddOp>(
              loc, type, determinant, product);

          sign = -sign;
        }
      }

      // The determinant should be completely folded into a constant since we
      // are operating only with constants.
      auto cast = mlir::dyn_cast<ConstantOp>(determinant.getDefiningOp());

      return cast.getValue();
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
      std::vector<mlir::Attribute>& constantVector,
      Equations<MatchedEquation> equations,
      mlir::OpBuilder& builder)
  {

    /// Get the filtered variables from one of the equations
    assert(equations.size() > 0);
    Variables variables = equations[0]->getVariables();

    /// For each equation, get the coefficients of the variables and the
    /// constant term, and save them respectively in the system matrix and
    /// constant term vector.
    for(size_t i = 0; i < equations.size(); ++i) {
      auto equation = equations[i].get();
      auto coefficients = std::vector<mlir::Attribute>();
      mlir::Attribute constantTerm;

      coefficients.resize(equations.size());

      auto res = equation->getCoefficients(builder, coefficients, constantTerm);

      if(mlir::failed(res)) {
        return false;
      }

      for(size_t j = 0; j < equations.size(); ++j) {
        matrix(i, j) = coefficients[j];
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
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto loc = builder.getUnknownLoc();
      auto type = RealAttr::get(builder.getContext(), 0).getType();

      bool res = false;

      /// Clone the system equations so that we can operate on them without
      /// disrupting the rest of the compilation process.
      Equations<MatchedEquation> equations;
      Equations<MatchedEquation> clones;

      equations = model.getEquations();

      for (const auto& equation : equations) {
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
      auto storage = std::vector<mlir::Attribute>();
      storage.resize(numberOfScalarEquations*numberOfScalarEquations);

      auto matrix = SquareMatrix(storage, numberOfScalarEquations);
      auto constantVector = std::vector<mlir::Attribute>(numberOfScalarEquations);

      /// Populate system coefficient matrix and constant term vector.
      res = getModelMatrixAndVector(matrix, constantVector, clones, builder);

      if(res) {
        auto solutionVector = std::vector<mlir::Attribute>();
        solutionVector.resize(numberOfScalarEquations);

        /// Create a temporary matrix to contain the matrices we are going to
        /// derive from the original one, by substituting the constant term
        /// vector to each of its columns.
        auto tempStorage = std::vector<mlir::Attribute>();
        tempStorage.resize(numberOfScalarEquations*numberOfScalarEquations);
        auto temp = SquareMatrix(tempStorage, numberOfScalarEquations);

        /// Check that the system matrix is non singular
        mlir::Attribute detA = matrix.det(builder);
        if(detA == 0) return false;

        /// Compute the determinant of each one of the matrices obtained by
        /// substituting the constant term vector to each one of the matrix
        /// columns, and divide them by the determinant of the system matrix.
        for (size_t i = 0; i < numberOfScalarEquations; i++)
        {
          matrix.substituteColumn(temp, i, constantVector);

          auto tempDet = temp.det(builder);

          auto tempDetOp = builder.create<ConstantOp>(
              loc, tempDet);

          auto detAOp = builder.create<ConstantOp>(
              loc, detA);

          auto div = builder.createOrFold<DivOp>(
              loc, type, tempDetOp, detAOp);

          //TODO fold if not already done
          auto divCast = mlir::dyn_cast<ConstantOp>(div.getDefiningOp());

          solutionVector[i] = divCast.getValue();
        }

        /// Set the results computed as the right side of the cloned equations,
        /// and the matched variables as the left side. Set the cloned equations
        /// as the model equations.
        //TODO: delete useless modelica instructions
        for(const auto& equation : clones) {

          auto access = equation->getWrite();
          auto& path = access.getPath();
          auto variable = access.getVariable();
          auto argument = variable->getValue().cast<mlir::BlockArgument>();
          auto offset = equation->getFlatAccessIndex(access, variable->getIndices());

          mlir::Attribute constant = solutionVector[argument.getArgNumber() + offset];

          equation->setMatchSolution(builder, constant);
        }

        model.setEquations(clones);
      }

      model.getOperation().dump();
      return res;
    }
  };
}