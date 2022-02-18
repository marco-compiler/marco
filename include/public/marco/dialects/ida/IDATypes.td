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

def VariablesListType : IDA_Type<"VariablesList", "vars_list"> {
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

def IndicesListType : IDA_Type<"IndicesList", "indices_list"> {
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let printer = [{
        $_printer << "indices_list";
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
