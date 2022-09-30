#include "marco/Codegen/Transforms/ModelSolving/CyclesCramerSolver.h"

using namespace ::marco::codegen;

SquareMatrix::SquareMatrix(
    std::vector<mlir::Value>& storage,
    size_t size) : size(size), storage(storage)
{
}

size_t SquareMatrix::getSize() const
{
  return size;
}

mlir::Value& SquareMatrix::operator()(size_t row, size_t col)
{
  return storage[row*size + col];
}

mlir::Value SquareMatrix::operator()(size_t row, size_t col) const
{
  return storage[row*size + col];
}

void SquareMatrix::dump()
{
  std::cerr << "Matrix size: " << getSize() << "\n";
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      (*this)(i,j).dump();
    }
    std::cerr << "\n";
  }
  std::cerr << "\n" << std::flush;
}

SquareMatrix SquareMatrix::subMatrix(SquareMatrix out, size_t row, size_t col)
{
  for (size_t i = 0; i < size; i++)
  {
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
    std::vector<mlir::Value>& col)
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

mlir::Value SquareMatrix::det(mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  SquareMatrix matrix = (*this);
  mlir::Value determinant;
  mlir::Location loc = builder.getUnknownLoc();


  if(size == 1)
    /// The matrix is a scalar, the determinant is the scalar itself
    determinant = matrix(0, 0);

  else if(size == 2) {
    /// The matrix is 2x2, the determinant is ad - bc where the matrix is:
    //    a b
    //    c d
    auto a = matrix(0,0);
    auto b = matrix(0,1);
    auto c = matrix(1,0);
    auto d = matrix(1,1);

    auto ad = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(a.getType(), d.getType()), a, d);
    auto bc = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(b.getType(), c.getType()), b, c);

    determinant = builder.createOrFold<SubOp>(
        loc,
        getMostGenericType(ad.getType(), bc.getType()), ad, bc);
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
    auto subMatrixStorage = std::vector<mlir::Value>(
        subMatrixSize*subMatrixSize);
    SquareMatrix out = SquareMatrix(
        subMatrixStorage, subMatrixSize);

    /// Initialize the determinant to zero.
    auto zeroAttr = RealAttr::get(
        builder.getContext(), 0);
    determinant = builder.create<ConstantOp>(
        loc, zeroAttr);

    /// For each scalar element of the first row of the matrix:
    for (size_t i = 0; i < size; i++) {
      auto scalarDet = matrix(0,i);

      auto matrixDet = subMatrix(out, 0, i).det(builder);

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
      determinant = builder.createOrFold<AddOp>(
          loc,
          getMostGenericType(determinant.getType(),
                             product.getType()),
          determinant, product);

      sign = -sign;
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
  auto storage = std::vector<mlir::Value>(storageSize);
  auto matrix = SquareMatrix(storage, numberOfScalarEquations);
  auto constantVector = std::vector<mlir::Value>(numberOfScalarEquations);

  /// Populate system coefficient matrix and constant term vector.
  res = getModelMatrixAndVector(
      matrix, constantVector, clones, builder);

  /// Populate an array containing the flat size of each possibly non scalar
  /// variable. This allows us to identify uniquely each equation and its
  /// matched variable.
  Variables variables = equations[0]->getVariables();
  std::vector<size_t> variableSizes(variables.size());
  getVariablesFlatSize(variableSizes, variables);

  if(res) {
    for(auto& equation : clones) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(
          equation->getOperation().bodyBlock(),
          equation->getOperation().bodyBlock()->begin());
      /// Allocate a new coefficient matrix and constant vector to contain the
      /// cloned ones.
      auto cloneStorage = std::vector<mlir::Value>(storageSize);
      auto clonedMatrix = SquareMatrix(cloneStorage, numberOfScalarEquations);
      auto clonedVector = std::vector<mlir::Value>(numberOfScalarEquations);

      clone(builder, clonedMatrix, matrix);

      /// Create a temporary matrix to contain the matrices we are going to
      /// derive from the original one, by substituting the constant term
      /// vector to each of its columns.
      auto tempStorage = std::vector<mlir::Value>();
      tempStorage.resize(
          numberOfScalarEquations*numberOfScalarEquations);
      auto temp = SquareMatrix(tempStorage, numberOfScalarEquations);

      /// Compute the determinant of the system matrix.
      mlir::Value determinant = clonedMatrix.det(builder);

      /// Get path, variable and argument number.
      auto access = equation->getWrite();
      auto& path = access.getPath();
      auto variable = access.getVariable();
      auto argument =
          variable->getValue().cast<mlir::BlockArgument>();

      /// Get flat access index, unique identifier of a scalar (ized) variable.
      auto offset = equation->getFlatAccessIndex(
          access, variable->getIndices());

      auto base = getSizeUntilVariable(
          argument.getArgNumber(), variableSizes);

      /// Compute the determinant of each one of the matrices obtained by
      /// substituting the constant term vector to each one of the matrix
      /// columns, and divide them by the determinant of the system matrix.
      clonedMatrix.substituteColumn(
          temp, base + offset, constantVector);

      auto tempDet = temp.det(builder);

      //TODO determine the type of a DivOp
      auto div = builder.createOrFold<DivOp>(
          loc,
          RealAttr::get(builder.getContext(), 0).getType(),
          tempDet, determinant);

      /// Set the results computed as the right side of the cloned equations,
      /// and the matched variables as the left side. Set the cloned equations
      /// as the model equations.

      equation->setMatchSolution(builder, div);
      model.setEquations(clones);
    }
  }

  return res;
}

bool CramerSolver::getModelMatrixAndVector(
    SquareMatrix matrix,
    std::vector<mlir::Value>& constantVector,
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
    auto coefficients = std::vector<mlir::Value>();
    mlir::Value constantTerm;

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

mlir::Value CramerSolver::cloneValueAndDependencies(
    mlir::OpBuilder& builder,
    mlir::Value value)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  std::stack<mlir::Operation*> visitStack;
  llvm::SmallVector<mlir::Operation*, 3> ops;

  /// Push the defining operation of the input value on the stack.
  if (auto definingOp = value.getDefiningOp()) {
    visitStack.push(definingOp);
  }

  /// Until the stack is empty pop it, add the operand to the list, get its
  /// operands (if any) and push them on the stack.
  while (!visitStack.empty()) {
    auto op = visitStack.top();
    visitStack.pop();

    ops.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        visitStack.push(definingOp);
      }
    }
  }

  size_t i;
  for (i = 0; i < ops.size(); ++i)
    builder.clone(*ops[i]);

  auto op = ops[i-1];
  auto results = op->getResults();
  assert(results.size() == 1);
  mlir::Value result = results[0];
  return result;
}

void CramerSolver::clone(mlir::OpBuilder& builder, SquareMatrix out, SquareMatrix in)
{
  auto size = in.getSize();
  for (size_t i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      out(i,j) = cloneValueAndDependencies(builder, in(i,j));
    }
  }
}

void CramerSolver::getVariablesFlatSize(
    std::vector<size_t>& variableSizes,
    Variables variables)
{
  for (int i = 0; i < variables.size(); ++i) {
    auto indices = *variables[i]->getIndices().rangesBegin();
    variableSizes[i] = indices.flatSize();
  }
}

size_t CramerSolver::getSizeUntilVariable(
    size_t index,
    std::vector<size_t>& variableSizes)
{
  assert(index <= variableSizes.size());

  size_t res = 0;
  for (size_t i = 0; i < index; ++i) {
    res += variableSizes[i];
  }
  return res;
}
