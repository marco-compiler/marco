#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica type definitions
//===----------------------------------------------------------------------===//

class Modelica_Type<string name>
    : TypeDef<Modelica_Dialect, name>;

def BooleanType : Modelica_Type<"Boolean"> {
    let mnemonic = "bool";
    let summary = "Modelica's Boolean type";

    let description = [{
        Type representing Modelica's Boolean.
    }];
}

def IntegerType : Modelica_Type<"Integer"> {
    let mnemonic = "int";
    let summary = "Modelica's Integer type";

    let description = [{
        Type representing Modelica's Integer.
    }];
}

def RealType : Modelica_Type<"Real"> {
    let mnemonic = "real";
    let summary = "Modelica's Real type";

    let description = [{
        Type representing Modelica's Real.
    }];
}

def ArrayType : Modelica_Type<"Array"> {
    let mnemonic = "array";
    let summary = "Array type";

    let description = [{
        Array type.
    }];

    let parameters = (ins
        "mlir::Type":$elementType,
        ArrayRefParameter<"long", "Dimensions">:$shape
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        static constexpr int64_t kDynamicSize = -1;

        unsigned int getRank() const;

        unsigned int getConstantDimensionsCount() const;
        unsigned int getDynamicDimensionsCount() const;

        bool hasConstantShape() const;

        long getFlatSize() const;

        bool isScalar() const;

        ArrayType slice(unsigned int subscriptsAmount) const;
        ArrayType toElementType(Type elementType) const;

		bool canBeOnStack() const;
     }];
}

def MemberType : Modelica_Type<"Member"> {
    let mnemonic = "member";
    let summary = "Member type";

    let description = [{
        Member type.
    }];

    let parameters = (ins
        "mlir::Type":$elementType,
        ArrayRefParameter<"long", "Dimensions">:$shape,
        "bool":$constantProperty,
        "IOProperty":$visibilityProperty
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        unsigned int getRank() const;

        bool isConstant() const {
            return getConstantProperty();
        }

        bool isInput() const {
            return getVisibilityProperty() == IOProperty::input;
        }

        bool isOutput() const {
            return getVisibilityProperty() == IOProperty::output;
        }

        static MemberType wrap(mlir::Type type, bool isConstant = false, IOProperty ioProperty = IOProperty::none);

        ArrayType toArrayType() const;
        mlir::Type unwrap() const;

        MemberType withShape(ArrayRef<long> shape) const;
        MemberType withType(Type type) const;
        MemberType withIOProperty(IOProperty ioProperty) const;
     }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
