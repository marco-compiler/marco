#ifndef MARCO_DIALECTS_IDA_IDATYPES_TD
#define MARCO_DIALECTS_IDA_IDATYPES_TD

include "marco/dialects/ida/IDADialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IDA type definitions
//===----------------------------------------------------------------------===//

class IDA_Type<string name, string typeMnemonic>
    : TypeDef<IDA_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def InstanceType : IDA_Type<"Instance", "instance"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "instance";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def VariablesType : IDA_Type<"Variables", "variables"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "vars_list";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def EquationType : IDA_Type<"Equation", "equation"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "equation";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def VariableType : IDA_Type<"Variable", "variable"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "variable";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def ResidualFunctionType : IDA_Type<"ResidualFunction", "residual_function"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "residual_function";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

def JacobianFunctionType : IDA_Type<"JacobianFunction", "jacobian_function"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "jacobian_function";
    }];

    let parser = [{
        return get($_ctxt);
    }];
}

#endif // MARCO_DIALECTS_IDA_IDATYPES_TD
