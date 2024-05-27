#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICATYPES_TD
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICATYPES_TD

include "marco/Dialect/BaseModelica/IR/BaseModelica.td"
include "marco/Dialect/BaseModelica/IR/BaseModelicaTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===---------------------------------------------------------------------===//
// Base Modelica type definitions
//===---------------------------------------------------------------------===//

// Base class for Base Modelica dialect types.
class BaseModelica_Type<
    string typeName,
    list<Trait> traits = [],
    string baseCppClass = "::mlir::Type">
    : TypeDef<BaseModelica_Dialect, typeName, traits, baseCppClass>
{
    let mnemonic = ?;
}

def BooleanType : BaseModelica_Type<"Boolean",
    [ElementaryType,
    MemRefElementTypeInterface]>
{
    let summary = "Boolean type";
    let mnemonic = "bool";

    let description = [{
        Type representing Modelica's Boolean.
    }];
}

def IntegerType : BaseModelica_Type<"Integer",
    [ElementaryType,
    MemRefElementTypeInterface]>
{
    let summary = "Integer type";
    let mnemonic = "int";

    let description = [{
        Type representing Modelica's Integer.
    }];
}

def RealType : BaseModelica_Type<"Real",
    [ElementaryType,
    MemRefElementTypeInterface]>
{
    let summary = "Real type";
    let mnemonic = "real";

    let description = [{
        Type representing Modelica's Real.
    }];
}

def ArrayType : BaseModelica_Type<"Array",
    [ShapedTypeInterface],
    "BaseArrayType">
{
    let summary = "Array type";
    let mnemonic = "array";

    let description = [{
        Type representing a Base Modelica array.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "mlir::Type":$elementType,
        "mlir::Attribute":$memorySpace
    );

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [
        TypeBuilderWithInferredContext<(ins
            "llvm::ArrayRef<int64_t>":$shape,
            "mlir::Type":$elementType,
            CArg<"Attribute", "{}">:$memorySpace)>
    ];

    let extraClassDeclaration = [{
        static constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;

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

def UnrankedArrayType : BaseModelica_Type<"UnrankedArray",
    [ShapedTypeInterface],
    "BaseArrayType">
{
    let summary = "Unranked array type";
    let mnemonic = "unranked_array";

    let description = [{
        TODO
    }];

    let parameters = (ins
        "mlir::Type":$elementType,
        "mlir::Attribute":$memorySpace
    );

    let hasCustomAssemblyFormat = 1;
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

        llvm::ArrayRef<int64_t> getShape() const { return std::nullopt; }
    }];

    let genVerifyDecl = 1;
}

def Modelica_RecordType : BaseModelica_Type<"Record">
{
    let summary = "Record type";
    let mnemonic = "record";

    let description = [{
        Record type.
    }];

    let parameters = (ins
        AttrParameter<"mlir::SymbolRefAttr", "Name">:$name
    );

    let assemblyFormat = "$name";

    let extraClassDeclaration = [{
        mlir::Operation* getRecordOp(
            mlir::SymbolTableCollection& symbolTable,
            mlir::ModuleOp moduleOp);
    }];
}

def VariableType : BaseModelica_Type<"Variable",
    [DeclareTypeInterfaceMethods<ShapedTypeInterface>]>
{
    let summary = "Variable type";
    let mnemonic = "variable";

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

    let hasCustomAssemblyFormat = 1;
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
        static constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;

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

        /// Get whether the type represents a scalar variable.
        bool isScalar() const;

        /// Get whether the variable has the 'discrete' modifier.
        bool isDiscrete() const;

        /// Get whether the variable has the 'parameter' modifier.
        bool isParameter() const;

        /// Get whether the variable has the 'constant' modifier.
        bool isConstant() const;

        /// Get whether the parameter is a parameter or a constant.
        bool isReadOnly() const;

        /// Get whether the variable has the 'input' modifier.
        bool isInput() const;

        /// Get whether the variable has the 'output' modifier.
        bool isOutput() const;

        static VariableType wrap(
            mlir::Type type,
            VariabilityProperty variabilityProperty =
                VariabilityProperty::none,
            IOProperty ioProperty = IOProperty::none);

        ArrayType toArrayType() const;
        mlir::TensorType toTensorType() const;
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

def EquationType : BaseModelica_Type<"Equation">
{
    let summary = "Equation type";
    let mnemonic = "equation";

    let description = [{
        Type representing a Base Modelica equation.
    }];
}

def RangeType : BaseModelica_Type<"Range", [IterableTypeInterface]>
{
    let summary = "Range type";
    let mnemonic = "range";

    let description = [{
        Type representing a range.
    }];

    let parameters = (ins
        "mlir::Type":$inductionType
    );

    let assemblyFormat = "$inductionType";
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICATYPES_TD
