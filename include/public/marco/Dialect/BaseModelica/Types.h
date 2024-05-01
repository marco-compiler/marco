#ifndef MARCO_DIALECTS_BASEMODELICA_TYPES_H
#define MARCO_DIALECTS_BASEMODELICA_TYPES_H

#include "marco/Dialect/BaseModelica/TypeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

namespace mlir::bmodelica
{
  enum class VariabilityProperty
  {
    none,
    discrete,
    parameter,
    constant
  };

  enum class IOProperty
  {
    input,
    output,
    none
  };

  //===-------------------------------------------------------------------===//
  // BaseArrayType
  //===-------------------------------------------------------------------===//

  /// This class provides a shared interface for ranked and unranked array types.
  class BaseArrayType
      : public mlir::Type, public mlir::ShapedType::Trait<BaseArrayType>
  {
    public:
      using mlir::Type::Type;

      /// Returns the element type of this array type.
      mlir::Type getElementType() const;

      /// Returns if this type is ranked, i.e. it has a known number of
      /// dimensions.
      bool hasRank() const;

      /// Returns the shape of this array type.
      llvm::ArrayRef<int64_t> getShape() const;

      /// Returns the memory space in which data referred to by this array
      /// resides.
      mlir::Attribute getMemorySpace() const;

      // TODO compare with MLIR repo
      /// Clone this type with the given shape and element type. If the
      /// provided shape is `None`, the current shape of the type is used.
      BaseArrayType cloneWith(
          std::optional<llvm::ArrayRef<int64_t>> shape,
          mlir::Type elementType) const;

      /// Return true if the specified element type is ok in a array.
      static bool isValidElementType(mlir::Type type);

      /// Methods for support type inquiry through isa, cast, and dyn_cast.
      static bool classof(mlir::Type type);

      /// Allow implicit conversion to ShapedType.
      operator mlir::ShapedType() const;
  };
}

#define GET_TYPEDEF_CLASSES
#include "marco/Dialect/BaseModelica/BaseModelicaTypes.h.inc"

namespace mlir::bmodelica
{
  //===-------------------------------------------------------------------===//
  // ArrayType
  //===-------------------------------------------------------------------===//

  /// This is a builder type that keeps local references to arguments.
  /// Arguments that are passed into the builder must outlive the builder.
  class ArrayType::Builder {
    public:
      // Build from another ArrayType.
      explicit Builder(ArrayType other)
          : shape(other.getShape()),
            elementType(other.getElementType()),
            memorySpace(other.getMemorySpace())
      {
      }

      // Build from scratch.
      Builder(llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
          : shape(shape), elementType(elementType)
      {
      }

      Builder& setShape(llvm::ArrayRef<int64_t> newShape)
      {
        shape = newShape;
        return *this;
      }

      Builder& setElementType(mlir::Type newElementType)
      {
        elementType = newElementType;
        return *this;
      }

      Builder& setMemorySpace(mlir::Attribute newMemorySpace)
      {
        memorySpace = newMemorySpace;
        return *this;
      }

      operator ArrayType()
      {
        return ArrayType::get(shape, elementType, memorySpace);
      }

    private:
      llvm::ArrayRef<int64_t> shape;
      mlir::Type elementType;
      mlir::Attribute memorySpace;
  };

  //===-------------------------------------------------------------------===//
  // VariableType
  //===-------------------------------------------------------------------===//

  /// This is a builder type that keeps local references to arguments.
  /// Arguments that are passed into the builder must outlive the builder.
  class VariableType::Builder
  {
    public:
      // Build from another VariableType.
      explicit Builder(VariableType other)
          : shape(other.getShape()),
            elementType(other.getElementType()),
            variabilityProperty(other.getVariabilityProperty()),
            visibilityProperty(other.getVisibilityProperty()),
            memorySpace(other.getMemorySpace())
      {
      }

      // Build from scratch.
      Builder(llvm::ArrayRef<int64_t> shape, mlir::Type elementType)
          : shape(shape), elementType(elementType)
      {
      }

      Builder& setShape(llvm::ArrayRef<int64_t> newShape)
      {
        shape = newShape;
        return *this;
      }

      Builder& setElementType(mlir::Type newElementType)
      {
        elementType = newElementType;
        return *this;
      }

      Builder& setVariabilityProperty(
          VariabilityProperty newVariabilityProperty)
      {
        variabilityProperty = newVariabilityProperty;
        return *this;
      }

      Builder& setVisibilityProperty(IOProperty newVisibilityProperty)
      {
        visibilityProperty = newVisibilityProperty;
        return *this;
      }

      Builder& setMemorySpace(mlir::Attribute newMemorySpace)
      {
        memorySpace = newMemorySpace;
        return *this;
      }

      operator VariableType()
      {
        return VariableType::get(
            shape,
            elementType,
            variabilityProperty,
            visibilityProperty,
            memorySpace);
      }

      operator ShapedType()
      {
        return VariableType::get(
            shape,
            elementType,
            variabilityProperty,
            visibilityProperty,
            memorySpace);
      }

    private:
      llvm::ArrayRef<int64_t> shape;
      mlir::Type elementType;
      VariabilityProperty variabilityProperty;
      IOProperty visibilityProperty;
      mlir::Attribute memorySpace;
  };
}

namespace mlir::bmodelica::detail
{
  /// Checks if the memorySpace has supported Attribute type.
  bool isSupportedMemorySpace(mlir::Attribute memorySpace);

  /// Replaces default memorySpace (integer == `0`) with empty Attribute.
  mlir::Attribute skipDefaultMemorySpace(mlir::Attribute memorySpace);
}

#endif // MARCO_DIALECTS_BASEMODELICA_TYPES_H
