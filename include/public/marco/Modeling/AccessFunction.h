#ifndef MARCO_MODELING_ACCESSFUNCTION_H
#define MARCO_MODELING_ACCESSFUNCTION_H

#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/Casting.h"
#include <variant>

namespace llvm {
class raw_ostream;
}

namespace marco::modeling {
/// The access function describes how an array variable is accessed.
class AccessFunction {
public:
  enum class Kind {
    Generic,
    Constant,
    Generic_LastArgument,
    Affine,
    Empty,
    RotoTranslation,
    Affine_LastArgument,
  };

  static std::unique_ptr<AccessFunction>
  build(mlir::MLIRContext *context, unsigned int numOfDimensions,
        llvm::ArrayRef<std::unique_ptr<DimensionAccess>> results);

  static std::unique_ptr<AccessFunction> build(mlir::AffineMap affineMap);

  static std::unique_ptr<AccessFunction>
  fromExtendedMap(mlir::AffineMap affineMap,
                  const DimensionAccess::FakeDimensionsMap &fakeDimensionsMap);

  static llvm::SmallVector<std::unique_ptr<DimensionAccess>>
  convertAffineExpressions(llvm::ArrayRef<mlir::AffineExpr> expressions);

protected:
  AccessFunction(Kind kind, mlir::MLIRContext *context);

public:
  AccessFunction(const AccessFunction &other);

  virtual ~AccessFunction();

  friend void swap(AccessFunction &first, AccessFunction &second);

  [[nodiscard]] virtual std::unique_ptr<AccessFunction> clone() const = 0;

  /// @name LLVM-style RTTI methods.
  /// {

  [[nodiscard]] Kind getKind() const { return kind; }

  template <typename T>
  [[nodiscard]] bool isa() const {
    return llvm::isa<T>(this);
  }

  template <typename T>
  [[nodiscard]] T *cast() {
    return llvm::cast<T>(this);
  }

  template <typename T>
  [[nodiscard]] const T *cast() const {
    return llvm::cast<T>(this);
  }

  template <typename T>
  [[nodiscard]] T *dyn_cast() {
    return llvm::dyn_cast<T>(this);
  }

  template <typename T>
  [[nodiscard]] const T *dyn_cast() const {
    return llvm::dyn_cast<T>(this);
  }

  /// }

  virtual bool operator==(const AccessFunction &other) const = 0;

  virtual bool operator!=(const AccessFunction &other) const = 0;

  virtual llvm::raw_ostream &dump(llvm::raw_ostream &os) const = 0;

  [[nodiscard]] mlir::MLIRContext *getContext() const;

  /// Get the number of dimensions.
  [[nodiscard]] virtual uint64_t getNumOfDims() const = 0;

  /// Get the number of results.
  [[nodiscard]] virtual uint64_t getNumOfResults() const = 0;

  /// Check whether the access function is an affine one.
  [[nodiscard]] virtual bool isAffine() const;

  /// Get the affine map representing the access function.
  [[nodiscard]] virtual mlir::AffineMap getAffineMap() const;

  /// Check if the access function has an identity layout, that is if the
  /// i-th dimension accesses the i-th induction variable with offset 0
  /// (e.g. [i0][i1][i2]).
  [[nodiscard]] virtual bool isIdentity() const = 0;

  /// Combine the access function with another one.
  ///
  /// Examples:
  ///   [i1 + 3][i0 - 2] * [i1 + 6][i0 - 1] = [i0 + 4][i1 + 2]
  ///   [i1 + 3][i1 - 2] * [5][i0 - 1] = [5][i1 + 2]
  ///   [5][i0 + 3] * [i0 + 1][i1 - 1] = [6][i0 + 2]
  [[nodiscard]] virtual std::unique_ptr<AccessFunction>
  combine(const AccessFunction &other) const;

  /// Check whether the access function is invertible.
  /// An access function is invertible if all the available induction
  /// variables are used.
  [[nodiscard]] virtual bool isInvertible() const;

  /// Get the inverse access (if possible).
  /// Returns nullptr if the inverse function can't be computed.
  [[nodiscard]] virtual std::unique_ptr<AccessFunction> inverse() const;

  /// Apply the access function to a point, in order to obtain the mapped
  /// indices.
  [[nodiscard]] virtual IndexSet map(const Point &point) const = 0;

  /// Apply the access function to an index set, in order to obtain the
  /// mapped indices.
  [[nodiscard]] virtual IndexSet map(const IndexSet &indices) const = 0;

  /// Apply the inverse of the access function to a set of indices.
  /// If the access function is not invertible, then the inverse indices
  /// are determined starting from a parent set.
  [[nodiscard]] virtual IndexSet
  inverseMap(const IndexSet &accessedIndices,
             const IndexSet &parentIndices) const;

  [[nodiscard]] virtual std::unique_ptr<AccessFunction>
  getWithGivenDimensions(uint64_t requestedDims) const;

  [[nodiscard]] virtual llvm::SmallVector<std::unique_ptr<DimensionAccess>, 6>
  getGeneralizedAccesses() const = 0;

  [[nodiscard]] virtual mlir::AffineMap getExtendedAffineMap(
      DimensionAccess::FakeDimensionsMap &fakeDimensionsMap) const = 0;

  [[nodiscard]] virtual bool
  isScalarIndependent(const AccessFunction &other,
                      const IndexSet &sourceIndices) const;

private:
  Kind kind;
  mlir::MLIRContext *context;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const AccessFunction &obj);
} // namespace marco::modeling

#endif // MARCO_MODELING_ACCESSFUNCTION_H
