#ifndef MARCO_DIALECTS_IDA_IDA_TD
#define MARCO_DIALECTS_IDA_IDA_TD

include "marco/dialects/ida/IDAAttributes.td"
include "marco/dialects/ida/IDADialect.td"
include "marco/dialects/ida/IDATypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// Opaque pointer type
def IDAInstance : Type<CPred<"$_self.isa<::mlir::ida::InstanceType>()">,
                         "instance", "::mlir::ida::InstanceType">;

//===----------------------------------------------------------------------===//
// IDA operation definitions
//===----------------------------------------------------------------------===//

class IDA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IDA_Dialect, mnemonic, traits>;

def IDA_InitOp : IDA_Op<"init",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_StepOp : IDA_Op<"step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_CreateOp : IDA_Op<"create",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        I64Attr:$scalarEquations,
        I64Attr:$vectorEquations,
        I64Attr:$vectorVariables);

    let results = (outs IDAInstance:$instance);

    let assemblyFormat = [{
        attr-dict `:` type($instance)
    }];
}

def IDA_FreeOp : IDA_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs I1:$result);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($result)
    }];
}

def IDA_AddTimeOp : IDA_Op<"add_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$startTime,
        F64Attr:$endTime,
        F64Attr:$timeStep);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_AddToleranceOp : IDA_Op<"add_tolerance",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$relativeTolerance,
        F64Attr:$absoluteTolerance);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}



#endif // MARCO_DIALECTS_IDA_IDA_TD
