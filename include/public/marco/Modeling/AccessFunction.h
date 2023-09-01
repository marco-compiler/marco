#ifndef MARCO_MODELING_ACCESSFUNCTION_H
#define MARCO_MODELING_ACCESSFUNCTION_H

#include "marco/Modeling/IndexSet.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/Casting.h"

namespace marco::modeling
{
  /// The access function describes how an array variable is accessed.
  class AccessFunction
  {
    public:
      enum Kind
      {
        Empty,
        Constant,
        Generic,
        RotoTranslation,
        ZeroDims,
        ZeroResults
      };

      static std::unique_ptr<AccessFunction> build(mlir::AffineMap affineMap);

      explicit AccessFunction(mlir::AffineMap affineMap);

    protected:
      AccessFunction(Kind kind, mlir::AffineMap affineMap);

    public:
      virtual ~AccessFunction();

      virtual std::unique_ptr<AccessFunction> clone() const;

      /// @name LLVM-style RTTI methods.
      /// {

      Kind getKind() const
      {
        return kind;
      }

      static bool classof(const AccessFunction* obj)
      {
        return obj->getKind() == Generic;
      }

      template<typename T>
      bool isa() const
      {
        return llvm::isa<T>(this);
      }

      template<typename T>
      T* cast()
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      const T* cast() const
      {
        return llvm::cast<T>(this);
      }

      template<typename T>
      T* dyn_cast()
      {
        return llvm::dyn_cast<T>(this);
      }

      template<typename T>
      const T* dyn_cast() const
      {
        return llvm::dyn_cast<T>(this);
      }

      /// }

      bool operator==(const AccessFunction& other) const;

      bool operator!=(const AccessFunction& other) const;

      mlir::AffineMap getAffineMap() const;

      /// Get the number of dimensions.
      virtual size_t getNumOfDims() const;

      /// Get the number of results.
      virtual size_t getNumOfResults() const;

      /// Check if the access function has an identity layout, that is if the
      /// i-th dimension accesses the i-th induction variable with offset 0
      /// (e.g. [i0][i1][i2]).
      virtual bool isIdentity() const;

      /// Combine the access function with another one.
      ///
      /// Examples:
      ///   [i1 + 3][i0 - 2] * [i1 + 6][i0 - 1] = [i0 + 4][i1 + 2]
      ///   [i1 + 3][i1 - 2] * [5][i0 - 1] = [5][i1 + 2]
      ///   [5][i0 + 3] * [i0 + 1][i1 - 1] = [6][i0 + 2]
      virtual std::unique_ptr<AccessFunction> combine(
          const AccessFunction& other) const;

      /// Check whether the access function is invertible.
      /// An access function is invertible if all the available induction
      /// variables are used.
      virtual bool isInvertible() const;

      /// Get the inverse access (if possible).
      /// Returns nullptr if the inverse function can't be computed.
      virtual std::unique_ptr<AccessFunction> inverse() const;

      /// Apply the access function to the equation indices, in order to obtain
      /// the accessed variable.
      virtual Point map(const Point& equationIndices) const;

      /// Apply the access function to an index set, in order to obtain the
      /// mapped indices.
      virtual IndexSet map(const IndexSet& indices) const;

      /// Apply the inverse of the access function to a set of indices.
      /// If the access function is not invertible, then the inverse indices
      /// are determined starting from a parent set.
      virtual IndexSet inverseMap(
          const IndexSet& accessedIndices,
          const IndexSet& parentIndices) const;

      virtual std::unique_ptr<AccessFunction> getWithAtLeastNDimensions(
          unsigned int dimensions) const;

    protected:
      mlir::AffineMap
      getAffineMapWithAtLeastNDimensions(unsigned int dimensions) const;

    private:
      Kind kind;
      mlir::AffineMap affineMap;
  };
}

#endif  // MARCO_MODELING_ACCESSFUNCTION_H
