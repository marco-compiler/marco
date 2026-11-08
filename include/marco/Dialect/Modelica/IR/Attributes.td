#ifndef MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_TD
#define MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_TD

include "marco/Dialect/Modelica/IR/Dialect.td"
include "marco/Dialect/Modelica/IR/AttrInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===---------------------------------------------------------------------===//
// Modelica attribute definitions
//===---------------------------------------------------------------------===//

class Modelica_Attr<string name,
                    list<Trait> traits = [],
                    string baseCppClass = "::mlir::Attribute">
    : AttrDef<Modelica_Dialect, name, traits, baseCppClass>
{
    let mnemonic = ?;
}

def Modelica_BooleanAttr : Modelica_Attr<"Boolean", [TypedAttrInterface]>
{
    let summary = "Boolean literal";
    let mnemonic = "bool";

    let description = [{
        Attribute representing a Boolean literal.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "bool":$value
    );

    let builders = [
        AttrBuilder<(ins "bool":$value), [{
            return $_get($_ctxt, BooleanType::get($_ctxt), value);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def Modelica_IntegerAttr : Modelica_Attr<"Integer", [TypedAttrInterface]>
{
    let summary = "Integer literal";
    let mnemonic = "int";

    let description = [{
        Attribute representing an Integer literal. The literal may have an
        attached unit, as allowed by unitful literals.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        "int64_t":$value,
        OptionalParameter<"::mlir::StringAttr">:$unit
    );

    let builders = [
        AttrBuilder<(ins "int64_t":$value), [{
            return $_get($_ctxt, IntegerType::get($_ctxt), value,
                         ::mlir::StringAttr());
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def Modelica_RealAttr : Modelica_Attr<"Real", [TypedAttrInterface]>
{
    let summary = "Real literal";
    let mnemonic = "real";

    let description = [{
        Attribute representing a Real literal. The literal may have an attached
        unit, as allowed by unitful literals.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        APFloatParameter<"">:$value,
        OptionalParameter<"::mlir::StringAttr">:$unit
    );

    let builders = [
        AttrBuilder<(ins "double":$value), [{
            return $_get($_ctxt, RealType::get($_ctxt), ::llvm::APFloat(value),
                         ::mlir::StringAttr());
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def Modelica_StringValueAttr
    : Modelica_Attr<"StringValue", [TypedAttrInterface]>
{
    let summary = "String literal";
    let mnemonic = "string";

    let description = [{
        Attribute representing a String literal.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        StringRefParameter<"String value">:$value    
    );

    let builders = [
        AttrBuilder<(ins "::llvm::StringRef":$value), [{
            return $_get($_ctxt, StringType::get($_ctxt), value);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def Modelica_EnumLiteralAttr
    : Modelica_Attr<"EnumLiteral", [TypedAttrInterface]>
{
    let summary = "Enumeration literal";
    let mnemonic = "enum";

    let description = [{
        Attribute representing a literal of an enumeration type. Besides its
        name, the literal stores its ordinal value, that is the value of
        `Integer(e)`, so that constant expressions involving enumerations can be
        evaluated without looking up the declaration of the enumeration type.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        StringRefParameter<"Literal name">:$literal,
        "int64_t":$ordinal
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

def Modelica_ArrayValueAttr
    : Modelica_Attr<"ArrayValue", [TypedAttrInterface]>
{
    let summary = "Constant array";
    let mnemonic = "array";

    let description = [{
        Attribute representing a constant array. The type of the attribute must
        be a statically shaped tensor, and the elements are stored in row-major
        order. Each element is a scalar attribute of the dialect, whose type
        matches the element type of the tensor.
    }];

    let parameters = (ins
        AttributeSelfTypeParameter<"">:$type,
        ArrayRefParameter<"::mlir::Attribute", "Elements">:$elements
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;

    let extraClassDeclaration = [{
        ::mlir::RankedTensorType getTensorType() const;
    }];
}

def Modelica_DimensionAttr : Modelica_Attr<"Dimension">
{
    let summary = "Array dimension of a declaration";
    let mnemonic = "dim";

    let description = [{
        Attribute describing an array dimension in a declaration. A dimension
        can be:
        - `:`, meaning that its size is unknown;
        - an integer literal, for dimensions given by a literal size;
        - `?`, meaning that its size is given by an Integer expression, computed
          by the region of the declaring operation;
        - `Boolean`, meaning that the dimension is indexed by Boolean values;
        - a symbol, meaning that the dimension is indexed by the literals of the
          referenced enumeration type.

        Example:

        ```mlir
        #modelica<dim :>
        #modelica<dim 3>
        #modelica<dim ?>
        #modelica<dim Boolean>
        #modelica<dim @ShirtSizes>
        ```
    }];

    let parameters = (ins
        "::mlir::modelica::DimensionKind":$kind,
        DefaultValuedParameter<"int64_t", "-1">:$size,
        OptionalParameter<"::mlir::SymbolRefAttr">:$enumeration
    );

    let builders = [
        AttrBuilder<(ins "::mlir::modelica::DimensionKind":$kind), [{
            return $_get($_ctxt, kind, -1, ::mlir::SymbolRefAttr());
        }]>,
        AttrBuilder<(ins "int64_t":$size), [{
            return $_get($_ctxt, ::mlir::modelica::DimensionKind::fixed, size,
                         ::mlir::SymbolRefAttr());
        }]>,
        AttrBuilder<(ins "::mlir::SymbolRefAttr":$enumeration), [{
            return $_get($_ctxt, ::mlir::modelica::DimensionKind::enumeration,
                         -1, enumeration);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif // MARCO_DIALECT_MODELICA_IR_ATTRIBUTES_TD
