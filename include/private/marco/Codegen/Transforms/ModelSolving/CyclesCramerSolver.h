
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

    /// Constructor of the matrix of size n.
    /// \param storage Vector that will contain the values of the matrix.
    /// Should be of size n*n.
    /// \param size Matrix size n.
    SquareMatrix(
      std::vector<mlir::Attribute>& storage,
      size_t size);

    /// Returns the matrix size n, where the matrix storage is n*n
    size_t getSize() const;

    /// Modify matrix elements with parentheses operator
    mlir::Attribute& operator()(size_t row, size_t col);

    /// Access matrix elements with parentheses operator
    mlir::Attribute operator()(size_t row, size_t col) const;

    /// Print the matrix contents to the specified stream.
    /// \param stream Stream to dump the matrix on.
    void print(llvm::raw_ostream &os);

    /// Print the matrix contents to the specified stream.
    void dump();

    /// The out matrix will contain the elements of the submatrix, minus the row
    /// and the column specified.
    /// \param out Output matrix.
    /// \param row Row to remove.
    /// \param col Column to remove.
    /// \return The out matrix.
    SquareMatrix subMatrix(SquareMatrix out, size_t row, size_t col);

    /// Substitute the column of the matrix indicated with the column number
    /// with the one specified by the col pointer, and return it in the out
    /// matrix given as input.
    /// \param out The matrix that will contain the result.
    /// \param colNumber The index of the matrix column we want to substitute.
    /// \param col The pointer to the vector that we want to substitute to the
    /// matrix column.
    /// \return The matrix obtained by substituting the given column in the
    /// current matrix at the specified column index
    SquareMatrix substituteColumn(
        SquareMatrix out,
        size_t colNumber,
        std::vector<mlir::Attribute>& col);

    /// Compute the determinant of the matrix
    /// \return The determinant of the matrix
    mlir::Attribute det(mlir::OpBuilder& builder);
  };

  class CramerSolver
  {
  private:
    mlir::OpBuilder& builder;
  public:
    /// The Solver class constructor.
    CramerSolver(mlir::OpBuilder& builder);

    /// Solve the system of equations contained in the model using the Cramer
    /// method, if such model is linear in the variables.
    /// \param model The model containing the system of linear equations.
    /// \return true if successful, false otherwise.
    bool solve(Model<MatchedEquation>& model);

    /// Get the matrix of the coefficients of the model and the constant vector
    /// \param matrix The matrix that will contain the model system coefficients.
    /// \param constantVector The vector that will contain the constant terms of
    /// the model equations.
    /// \param equations Cloned equations.
    /// \param builder The MLIR buillder.
    /// \return true if successful, false otherwise.
    static bool getModelMatrixAndVector(
        SquareMatrix matrix,
        std::vector<mlir::Attribute>& constantVector,
        Equations<MatchedEquation> equations,
        mlir::OpBuilder& builder);

    /// Given a set of variables compute their flat sizes.
    /// \param variableSizes Array to be filled with the size of each variable.
    /// \param variables Set of variables.
    static void getVariablesFlatSize(
        std::vector<size_t>& variableSizes,
        Variables variables);

    /// Get the value of the summed variable sizes until before the specified index.
    /// \param index Index of the variable to stop summing sizes.
    /// \param variableSizes Array containing the size of each variable.
    /// \return Sum of sizes before specified variable index.
    static size_t getSizeUntilVariable(
        size_t index,
        std::vector<size_t>& variableSizes);
  };
}