#ifndef MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
#define MARCO_DIALECTS_MODELICA_MODELICATYPES_TD

include "marco/Dialect/Modelica/ModelicaDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Modelica type definitions
//===----------------------------------------------------------------------===//

class Modelica_Type<string name, string typeMnemonic>
    : TypeDef<Modelica_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def BooleanType : Modelica_Type<"Boolean", "bool"> {
    let summary = "Modelica's Boolean type";

    let description = [{
        Type representing Modelica's Boolean.
    }];

    let printer = [{
        $_printer << "bool";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def IntegerType : Modelica_Type<"Integer", "int"> {
    let summary = "Modelica's Integer type";

    let description = [{
        Type representing Modelica's Integer.
    }];

    let printer = [{
        $_printer << "int";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def RealType : Modelica_Type<"Real", "real"> {
    let summary = "Modelica's Real type";

    let description = [{
        Type representing Modelica's Real.
    }];

    let printer = [{
        $_printer << "real";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def ArrayType : Modelica_Type<"Array", "array"> {
    let summary = "Array type";

    let description = [{
        Array type.
    }];

    let parameters = (ins
        "mlir::Type":$elementType,
        ArrayRefParameter<"long", "Dimensions">:$shape
    );

    let extraClassDeclaration = [{
        unsigned int getRank() const;

        unsigned int getConstantDimensionsCount() const;
        unsigned int getDynamicDimensionsCount() const;

        bool hasConstantShape() const;

        long getFlatSize() const;

        bool isScalar() const;

        ArrayType slice(unsigned int subscriptsAmount) const;
        ArrayType toElementType(mlir::Type elementType) const;
        UnsizedArrayType toUnsized() const;

		bool canBeOnStack() const;
     }];
}

def UnsizedArrayType : Modelica_Type<"UnsizedArray", "array"> {
    let summary = "Unsized array type";

    let description = [{
        Unsized rray type.
    }];

    let parameters = (ins "mlir::Type":$elementType);
}

def MemberType : Modelica_Type<"Member", "member"> {
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

    let extraClassDeclaration = [{
		public:
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

            MemberType withIOProperty(IOProperty ioProperty) const;
     }];
}

#endif // MARCO_DIALECTS_MODELICA_MODELICATYPES_TD
