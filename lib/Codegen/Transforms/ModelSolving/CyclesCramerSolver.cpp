#include "marco/Codegen/Transforms/ModelSolving/CyclesCramerSolver.h"

using namespace ::marco::codegen;

#define DEBUG_TYPE "CyclesSolving"

SquareMatrix::SquareMatrix(
    std::vector<mlir::Attribute>& storage,
    size_t size) : size(size), storage(storage)
{
}

size_t SquareMatrix::getSize() const
{
  return size;
}

mlir::Attribute& SquareMatrix::operator()(size_t row, size_t col)
{
  return storage[row*size + col];
}

mlir::Attribute SquareMatrix::operator()(size_t row, size_t col) const
{
  return storage[row*size + col];
}

void SquareMatrix::print(llvm::raw_ostream &os)
{
  os << "MATRIX SIZE: " << size << "\n";
  for (size_t i = 0; i < size; i++) {
    os << "LINE #" << i << "\n";
    for (size_t j = 0; j < size; j++) {
      (*this)(i,j).print(os);
      os << "\n";
    }
    os << "\n";
  }
  os << "\n";
  os.flush();
}

void SquareMatrix::dump()
{
  print(llvm::dbgs());
}

SquareMatrix SquareMatrix::subMatrix(SquareMatrix out, size_t row, size_t col)
{
  for (size_t i = 0; i < size; i++) {
    if(i != row) {
      for (size_t j = 0; j < size; j++) {
        if(j != col) {
          out(i < row ? i : i - 1, j < col ? j : j - 1) =
              (*this)(i, j);
        }
      }
    }
  }
  return out;
}

SquareMatrix SquareMatrix::substituteColumn(
    SquareMatrix out,
    size_t colNumber,
    std::vector<mlir::Attribute>& col)
{
  assert(colNumber < size && out.getSize() == size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if(j != colNumber)
        out(i,j) = (*this)(i,j);
      else
        out(i,j) = col[i];
    }
  }
  return out;
}

mlir::Attribute SquareMatrix::det(mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  SquareMatrix matrix = (*this);
  mlir::Attribute determinant;
  mlir::Location loc = builder.getUnknownLoc();


  if(size == 1)
    /// The matrix is a scalar, the determinant is the scalar itself
    determinant = matrix(0, 0);

  else if(size == 2) {
    /// The matrix is 2x2, the determinant is ad - bc where the matrix is:
    //    a b
    //    c d

    auto a = builder.create<ConstantOp>(loc, matrix(0,0));
    auto b = builder.create<ConstantOp>(loc, matrix(0,1));
    auto c = builder.create<ConstantOp>(loc, matrix(1,0));
    auto d = builder.create<ConstantOp>(loc, matrix(1,1));

    auto ad = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(a.getType(), d.getType()), a, d);
    auto bc = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(b.getType(), c.getType()), b, c);

    auto subOp = builder.createOrFold<SubOp>(
        loc,
        getMostGenericType(ad.getType(), bc.getType()), ad, bc);

    determinant = mlir::dyn_cast<ConstantOp>(subOp.getDefiningOp()).getValue();
  }
  else if (size > 2) {
    /// The matrix is 3x3 or greater. Compute the determinant by taking the
    /// first row of the matrix, and multiplying each scalar element of that
    /// row with the determinant of the submatrix corresponding to that
    /// element, that is the matrix constructed by removing from the
    /// original one the row and the column of the scalar element.
    /// Then add the ones in even position and subtract the ones in odd
    /// (counting from zero).
    int sign = 1;
    size_t subMatrixSize = size - 1;

    /// Create the matrix that will hold the n submatrices.
    auto subMatrixStorage = std::vector<mlir::Attribute>(
        subMatrixSize*subMatrixSize);
    SquareMatrix out = SquareMatrix(
        subMatrixStorage, subMatrixSize);

    /// Initialize the determinant to zero.
    auto zeroAttr = RealAttr::get(
        builder.getContext(), 0);
    mlir::Value tot = builder.create<ConstantOp>(
        loc, zeroAttr);

    /// For each scalar element of the first row of the matrix:
    for (size_t i = 0; i < size; i++) {
      auto scalarDet = builder.create<ConstantOp>(loc, matrix(0,i));
      auto matrixDet = builder.create<ConstantOp>(loc, subMatrix(out, 0, i).det(builder));

      /// Multiply the scalar element with the submatrix determinant.
      mlir::Value product = builder.createOrFold<MulOp>(
          loc,
          getMostGenericType(scalarDet.getType(),
                             matrixDet.getType()),
          scalarDet, matrixDet);

      /// If in odd position, negate the value.
      if(sign == -1)
        product = builder.createOrFold<NegateOp>(
            loc, product.getType(), product);

      /// Add the result to the rest of the determinant.
      tot = builder.createOrFold<AddOp>(
          loc,
          getMostGenericType(tot.getType(),
                             product.getType()),
          tot, product);

      sign = -sign;
      determinant = mlir::dyn_cast<ConstantOp>(tot.getDefiningOp()).getValue();
    }
  }

  return determinant;
}

//extract coefficients and constant terms from equations
//collect them into the system matrix and constant term vector

//for each eq in equations
//  set insertion point to beginning of eq
//  clone the system matrix inside of eq
//  clone the constant vector inside of eq
//  compute the solution with cramer
CramerSolver::CramerSolver(mlir::OpBuilder& builder) : builder(builder)
{
}

bool CramerSolver::solve(Model<MatchedEquation>& model)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto loc = builder.getUnknownLoc();

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
  auto storageSize = numberOfScalarEquations*numberOfScalarEquations;
  auto storage = std::vector<mlir::Attribute>(storageSize);
  auto matrix = SquareMatrix(storage, numberOfScalarEquations);
  auto constantVector = std::vector<mlir::Attribute>(numberOfScalarEquations);

  LLVM_DEBUG({
    llvm::dbgs() << "Populating the model matrix and constant vector.\n";
  });
  /// Populate system coefficient matrix and constant term vector.
  res = getModelMatrixAndVector(
      matrix, constantVector, clones, builder);

  LLVM_DEBUG({
    llvm::dbgs() << "COEFFICIENT MATRIX: \n";
    matrix.dump();
    llvm::dbgs() << "CONSTANT VECTOR: \n";
    for(auto el : constantVector)
      el.dump();

    model.getOperation().dump();
  });

  if(res) {
    for(auto& equation : clones) {
      mlir::OpBuilder::InsertionGuard equationGuard(builder);
      builder.setInsertionPoint(equation->getOperation().bodyBlock()->getTerminator());

      /// Create a temporary matrix to contain the matrices we are going to
      /// derive from the original one, by substituting the constant term
      /// vector to each of its columns.
      auto tempStorage = std::vector<mlir::Attribute>(numberOfScalarEquations*numberOfScalarEquations);
      auto temp = SquareMatrix(tempStorage, numberOfScalarEquations);

      /// Compute the determinant of the system matrix.
      mlir::Attribute determinant = matrix.det(builder);

      /// Get path, variable and argument.
      auto access = equation->getWrite();
      auto& path = access.getPath();

      /// Get flat access index, unique identifier of a scalar (ized) variable.
      auto index = equation->getFlatAccessIndex(
          access,equation->getIterationRanges());

      /// Compute the determinant of each one of the matrices obtained by
      /// substituting the constant term vector to each one of the matrix
      /// columns, and divide them by the determinant of the system matrix.
      matrix.substituteColumn(
          temp, index, constantVector);

      auto tempDet = temp.det(builder);

      //TODO determine the type of a DivOp
      auto div = builder.createOrFold<DivOp>(
          loc,
          RealAttr::get(builder.getContext(), 0).getType(),
          builder.create<ConstantOp>(loc, tempDet),
          builder.create<ConstantOp>(loc, determinant));

      /// Set the results computed as the right side of the cloned equations,
      /// and the matched variables as the left side. Set the cloned equations
      /// as the model equations.

      equation->setMatchSolution(builder, div);
    }
    LLVM_DEBUG({
        llvm::dbgs() << "MODEL: \n";
        model.getOperation()->dump();
    });

    model.setEquations(clones);
  }

  return res;
}

bool CramerSolver::getModelMatrixAndVector(
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

    auto res = equation->getCoefficients(
        builder, coefficients, constantTerm);

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
