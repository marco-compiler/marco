#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

//===----------------------------------------------------------------------===//
// Modelica type definitions
//===----------------------------------------------------------------------===//

// Base class for Modelica dialect types.
class Modelica_Type<
    string name,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<Modelica_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

def BooleanType : Modelica_Type<"Boolean">
{
    let summary = "Boolean type";
    let mnemonic = "bool";

    let description = [{
        Type representing Modelica's Boolean.
    }];
}

def IntegerType : Modelica_Type<"Integer">
{
    let summary = "Integer type";
    let mnemonic = "int";

    let description = [{
        Type representing Modelica's Integer.
    }];
}

def RealType : Modelica_Type<"Real">
{
    let summary = "Real type";
    let mnemonic = "real";

    let description = [{
        Type representing Modelica's Real.
    }];
}

def ArrayType : Modelica_Type<"Array",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>,
    ShapedTypeInterface],
    "BaseArrayType">
{
    let summary = "Array type";

    let description = [{
        Type representing a Modelica array.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::Attribute":$memorySpace
    );

    let skipDefaultBuilders = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "mlir::Type":$elementType,
            CArg<"Attribute", "{}">:$memorySpace)>
    ];

    let extraClassDeclaration = [{
        static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

        using ShapedType::Trait<ArrayType>::clone;
        using ShapedType::Trait<ArrayType>::getElementTypeBitWidth;
        using ShapedType::Trait<ArrayType>::getRank;
        using ShapedType::Trait<ArrayType>::getNumElements;
        using ShapedType::Trait<ArrayType>::isDynamicDim;
        using ShapedType::Trait<ArrayType>::hasStaticShape;
        using ShapedType::Trait<ArrayType>::getNumDynamicDims;
        using ShapedType::Trait<ArrayType>::getDimSize;
        using ShapedType::Trait<ArrayType>::getDynamicDimIndex;

        /// This is a builder type that keeps local references to arguments.
        /// Arguments that are passed into the builder must outlive the builder.
        class Builder;

        bool isScalar() const;

        ArrayType slice(unsigned int subscriptsAmount) const;
        ArrayType toElementType(Type elementType) const;
        ArrayType withShape(llvm::ArrayRef<int64_t> shape) const;

        bool canBeOnStack() const;
    }];

    let genVerifyDecl = 1;
}

def UnrankedArrayType : Modelica_Type<"UnrankedArray",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>,
    ShapedTypeInterface],
    "BaseArrayType">
{
    let summary = "Unranked array type";

    let description = [{
        TODO
    }];

    let parameters = (ins
        "mlir::Type":$elementType,
        "mlir::Attribute":$memorySpace
    );

    let skipDefaultBuilders = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType, "mlir::Attribute":$memorySpace), [{
            // Drop default memory space value and replace it with empty attribute.
            Attribute nonDefaultMemorySpace = skipDefaultMemorySpace(memorySpace);
            return $_get(elementType.getContext(), elementType, nonDefaultMemorySpace);
        }]>
    ];

    let extraClassDeclaration = [{
        using ShapedType::Trait<UnrankedArrayType>::clone;
        using ShapedType::Trait<UnrankedArrayType>::getElementTypeBitWidth;
        using ShapedType::Trait<UnrankedArrayType>::getRank;
        using ShapedType::Trait<UnrankedArrayType>::getNumElements;
        using ShapedType::Trait<UnrankedArrayType>::isDynamicDim;
        using ShapedType::Trait<UnrankedArrayType>::hasStaticShape;
        using ShapedType::Trait<UnrankedArrayType>::getNumDynamicDims;
        using ShapedType::Trait<UnrankedArrayType>::getDimSize;
        using ShapedType::Trait<UnrankedArrayType>::getDynamicDimIndex;

        llvm::ArrayRef<int64_t> getShape() const { return llvm::None; }
    }];

    let genVerifyDecl = 1;
}

def VariableType : Modelica_Type<"Variable",
    [DeclareTypeInterfaceMethods<SubElementTypeInterface>,
    DeclareTypeInterfaceMethods<ShapedTypeInterface>]>
{
    let summary = "Variable type";

    let description = [{
        Variable type.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "VariabilityProperty":$variabilityProperty,
        "IOProperty":$visibilityProperty,
        "mlir::Attribute":$memorySpace
    );

    let skipDefaultBuilders = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "mlir::Type":$elementType,
            "VariabilityProperty":$variabilityProperty,
            "IOProperty":$visibilityProperty,
            CArg<"Attribute", "{}">:$memorySpace)>
    ];

    let extraClassDeclaration = [{
        static constexpr int64_t kDynamicSize = mlir::ShapedType::kDynamicSize;

        static bool isValidElementType(mlir::Type type);

        using ShapedType::Trait<VariableType>::clone;
        using ShapedType::Trait<VariableType>::getElementTypeBitWidth;
        using ShapedType::Trait<VariableType>::getRank;
        using ShapedType::Trait<VariableType>::getNumElements;
        using ShapedType::Trait<VariableType>::isDynamicDim;
        using ShapedType::Trait<VariableType>::hasStaticShape;
        using ShapedType::Trait<VariableType>::getNumDynamicDims;
        using ShapedType::Trait<VariableType>::getDimSize;
        using ShapedType::Trait<VariableType>::getDynamicDimIndex;

        /// This is a builder type that keeps local references to arguments.
        /// Arguments that are passed into the builder must outlive the builder.
        class Builder;

        bool isScalar() const
        {
            return getRank() == 0;
        }

        bool isDiscrete() const
        {
            return getVariabilityProperty() == VariabilityProperty::discrete;
        }

        bool isParameter() const
        {
            return getVariabilityProperty() == VariabilityProperty::parameter;
        }

        bool isConstant() const
        {
            return getVariabilityProperty() == VariabilityProperty::constant;
        }

        bool isReadOnly() const
        {
            return isParameter() || isConstant();
        }

        bool isInput() const
        {
            return getVisibilityProperty() == IOProperty::input;
        }

        bool isOutput() const
        {
            return getVisibilityProperty() == IOProperty::output;
        }

        static VariableType wrap(
            mlir::Type type,
            VariabilityProperty variabilityProperty =
                VariabilityProperty::none,
            IOProperty ioProperty = IOProperty::none);

        ArrayType toArrayType() const;
        mlir::Type unwrap() const;

        VariableType withShape(llvm::ArrayRef<int64_t> shape) const;
        VariableType withType(mlir::Type type) const;

        VariableType withVariabilityProperty(
            VariabilityProperty variabilityProperty) const;

        VariableType withoutVariabilityProperty() const;
        VariableType asDiscrete() const;
        VariableType asParameter() const;
        VariableType asConstant() const;

        VariableType withIOProperty(IOProperty ioProperty) const;

        VariableType withoutIOProperty() const;
        VariableType asInput() const;
        VariableType asOutput() const;
    }];

    let genVerifyDecl = 1;
}

def Modelica_RecordType : Modelica_Type<"Record", []>
{
    let summary = "Record type";
    let mnemonic = "record";

    let description = [{
        Record type.
    }];

    let parameters = (ins
        AttrParameter<"mlir::SymbolRefAttr", "Name">:$name
    );

    let assemblyFormat = "`<` $name `>`";
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
