#include "marco/Codegen/Transforms/ModelSolving/CyclesCramerSolver.h"

using namespace ::marco::codegen;

#define DEBUG_TYPE "CyclesSolving"

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

bool CramerSolver::solve(std::vector<MatchedEquation>& equations)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto loc = builder.getUnknownLoc();

  bool res = false;

  /// Clone the system equations so that we can operate on them without
  /// disrupting the rest of the compilation process.
  Equations<MatchedEquation> clones;

  for (const auto& equation : equations) {
    auto clone = equation.clone();

    auto matchedClone = std::make_unique<MatchedEquation>(
        std::move(clone),
        equation.getIterationRanges(),
        equation.getWrite().getPath());

    clones.add(std::move(matchedClone));
  }

  /// Get the number of scalar equations in the system, which should be
  /// equal to the number of filtered variables.
  size_t systemSize = 0;
  for (const auto& equation : clones) {
    systemSize += equation->getIterationRanges().flatSize();
  }

  /// Create the matrix to contain the coefficients of the system's
  /// equations and the vector to contain the constant terms.
  auto storageSize = systemSize * systemSize;
  auto storage = std::vector<mlir::Value>(storageSize);
  auto matrix = SquareMatrix(storage, systemSize);
  auto constantVector = std::vector<mlir::Value>(systemSize);

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
  });

  /// Compute the mapping between the flat index and the local index.
  /// The map is sorted by key, in this case the flat index.
  // e.g.:
  // index flat    flat index
  // 0     1       0    1
  // 1     0    -> 1    0
  // 2     2       2    2
  std::vector<size_t> flatSizes;
  std::map<size_t, size_t> flatMap;
  for (size_t i = 0; i < clones.size(); ++i) {
    auto index = clones[i]->getFlatAccessIndex();
    flatSizes.push_back(index);
    flatMap.emplace(index, i);
  }

  /// Compute the mapping between the local index and the continuous index.
  /// The continuous index is sorted as the flat index but has no jumps.
  // e.g.:
  // flat  index   index continuous
  // 0     1       0     1
  // 1     0    -> 1     0
  // 2     2       2     2
  std::map<size_t, size_t> continuousMap;
  size_t counter = 0;
  for (auto it = flatMap.begin(); it != flatMap.end(); ++it) {
    continuousMap.emplace((*it).second, counter);
    ++counter;
  }

  if(res) {
    for(size_t i = 0; i < clones.size(); ++i) {
      auto& equation = clones[i];
      mlir::OpBuilder::InsertionGuard equationGuard(builder);
      builder.setInsertionPoint(equation->getOperation().bodyBlock()->getTerminator());
      /// Allocate a new coefficient matrix and constant vector to contain the
      /// cloned ones.
      auto cloneStorage = std::vector<mlir::Value>(storageSize);
      auto clonedMatrix = SquareMatrix(cloneStorage, systemSize);
      auto clonedVector = std::vector<mlir::Value>(systemSize);

      mlir::BlockAndValueMapping mapping;
      for (const auto& variable : equation->getVariables()) {
        auto variableValue = variable->getValue();
        mapping.map(variableValue, variableValue);
      }

      cloneCoefficientMatrix(builder, clonedMatrix, matrix, mapping);
      cloneConstantVector(builder, clonedVector, constantVector, mapping);

      /// Create a temporary matrix to contain the matrices we are going to
      /// derive from the original one, by substituting the constant term
      /// vector to each of its columns.
      auto tempStorage = std::vector<mlir::Value>();
      tempStorage.resize(
          systemSize * systemSize);
      auto temp = SquareMatrix(tempStorage, systemSize);

      /// Compute the determinant of the system matrix.
      mlir::Value determinant = clonedMatrix.det(builder);

      /// Get path, variable and argument number.
      auto access = equation->getWrite();
      auto& path = access.getPath();

      /// Compute the determinant of each one of the matrices obtained by
      /// substituting the constant term vector to each one of the matrix
      /// columns, and divide them by the determinant of the system matrix.
      clonedMatrix.substituteColumn(
          temp, continuousMap[i], clonedVector);

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
    }

    for (auto& equation : clones) {
      solution.add(std::move(equation));
    }
  } else {
    for (auto& equation : clones) {
      unsolved.add(std::move(equation));
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
  /// For each equation, get the coefficients of the variables and the
  /// constant term, and save them respectively in the system matrix and
  /// constant term vector.
  for(size_t i = 0; i < equations.size(); ++i) {
    auto equation = equations[i].get();
/*    for (const auto& range : *equation->getIterationRanges().rangesBegin()) {

    }*/

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
    mlir::Value value,
    mlir::BlockAndValueMapping& mapping)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  std::stack<mlir::Operation*> cloneStack;
  std::vector<mlir::Operation*> toBeCloned;

  /// Push the defining operation of the input value on the stack.
  if (auto op = value.getDefiningOp(); op != nullptr) {
    cloneStack.push(op);
  }

  /// Until the stack is empty pop it, add the operand to the list, get its
  /// operands (if any) and push them on the stack.
  while (!cloneStack.empty()) {
    auto op = cloneStack.top();
    cloneStack.pop();

    toBeCloned.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
        cloneStack.push(operandOp);
      }
    }
  }

  // Clone the operations
  mlir::Operation* clonedOp = nullptr;
  for (auto it = toBeCloned.rbegin(); it != toBeCloned.rend(); ++it) {
    mlir::Operation* op = *it;
    clonedOp = builder.clone(*op, mapping);
  }

  assert(clonedOp != nullptr);
  auto op = clonedOp;
  auto results = op->getResults();
  assert(results.size() == 1);
  mlir::Value result = results[0];
  return result;
}

void CramerSolver::cloneCoefficientMatrix(
    mlir::OpBuilder& builder,
    SquareMatrix out,
    SquareMatrix in,
    mlir::BlockAndValueMapping& mapping)
{
  auto size = in.getSize();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      out(i,j) = cloneValueAndDependencies(builder, in(i,j), mapping);
    }
  }
}

void CramerSolver::cloneConstantVector(
    mlir::OpBuilder& builder,
    std::vector<mlir::Value>& out,
    std::vector<mlir::Value>& in,
    mlir::BlockAndValueMapping& mapping)
{
  for (size_t i = 0; i < in.size(); ++i) {
      out[i] = cloneValueAndDependencies(builder, in[i], mapping);
  }
}

bool CramerSolver::hasUnsolvedCycles() const
{
  return unsolved.size() != 0;
}

Equations<MatchedEquation> CramerSolver::getSolution() const
{
  return solution;
}

Equations<MatchedEquation> CramerSolver::getUnsolvedEquations() const
{
  return unsolved;
}
