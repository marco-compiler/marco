#ifndef MARCO_DIALECTS_IDA_IDA_TD
#define MARCO_DIALECTS_IDA_IDA_TD

include "marco/dialects/ida/IDAAttributes.td"
include "marco/dialects/ida/IDADialect.td"
include "marco/dialects/ida/IDATypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// IDA instance type
def IDAInstance : Type<CPred<"$_self.isa<::mlir::ida::InstanceType>()">,
                       "instance", "::mlir::ida::InstanceType">;

// IDA variables list type
def IDAVariablesList : Type<CPred<"$_self.isa<::mlir::ida::VariablesListType>()">,
                                  "variables_list", "::mlir::ida::VariablesListType">;

// IDA residual function type
def IDAResidualFunction : Type<CPred<"$_self.isa<::mlir::ida::ResidualFunctionType>()">,
                               "residual_function", "::mlir::ida::ResidualFunctionType">;

// IDA Jacobian function type
def IDAJacobianFunction : Type<CPred<"$_self.isa<::mlir::ida::JacobianFunctionType>()">,
                               "jacobian_function", "::mlir::ida::JacobianFunctionType">;

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

def IDA_SetTimesOp : IDA_Op<"set_times",
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

def IDA_SetToleranceOp : IDA_Op<"set_tolerance",
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

def IDA_GetCurrentTimeOp : IDA_Op<"get_current_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs F64:$time);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($time)
    }];
}

def IDA_AddEquationOp : IDA_Op<"add_equation",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
        Explain 2xN
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64MemRef:$equationRanges);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equationRanges attr-dict `:` type($instance) `,` type($equationRanges)
    }];
}

def IDA_AddResidualOp : IDA_Op<"add_residual",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        SymbolNameAttr:$residualFunction);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_AddJacobianOp : IDA_Op<"add_jacobian",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        SymbolNameAttr:$jacobianFunction);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_AddVariableOp : IDA_Op<"add_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64Attr:$vectorVariableIndex,
        I64MemRef:$arrayDimensions,
        I1Attr:$isState);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $arrayDimensions attr-dict `:` type($instance) `,` type($arrayDimensions)
    }];
}

def IDA_AddVarAccessOp : IDA_Op<"add_var_access",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64Attr:$vectorVariableIndex,
        AffineMapAttr:$accessFunction);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_GetVariablesListOp : IDA_Op<"get_variables_list",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs IDAVariablesList:$variables);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($variables)
    }];
}

def IDA_GetDerivativesListOp : IDA_Op<"get_ders_list",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs IDAVariablesList:$ders);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($ders)
    }];
}

def IDA_GetVariableOp : IDA_Op<"get_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        IDAInstance:$instance,
        I64Attr:$scalarOffset);

    let results = (outs AnyType:$vectorVariable);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($vectorVariable)
    }];
}

def IDA_GetTimeOp : IDA_Op<"get_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs F64:$time);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($time)
    }];
}

def IDA_ResidualFunctionOp : IDA_Op<"residual_function",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 4> args;
            args.push_back($_builder.getF64Type());
            args.push_back(VariablesListType::get($_builder.getContext()));
            args.push_back(VariablesListType::get($_builder.getContext()));
            args.push_back(IndicesListType::get($_builder.getContext()));

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            //===------------------------------------------------------------------===//
            // FunctionOpInterface Methods
            //===------------------------------------------------------------------===//

            /// Returns the argument types of this function.
            ArrayRef<Type> getArgumentTypes() {
                return getType().getInputs();
            }

            /// Returns the result types of this function.
            ArrayRef<Type> getResultTypes() {
                return getType().getResults();
            }

            //===------------------------------------------------------------------===//
            // SymbolOpInterface Methods
            //===------------------------------------------------------------------===//

            bool isDeclaration() {
                return isExternal();
            }

        private:
            // This trait needs access to the hooks defined below.
            friend class OpTrait::FunctionLike<ResidualFunctionOp>;

            /// Returns the number of arguments. This is a hook for
            /// OpTrait::FunctionLike.
            unsigned getNumFuncArguments() {
                return getType().getInputs().size();
            }

            /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
            unsigned getNumFuncResults() {
                return getType().getResults().size();
            }

            /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
            /// attribute is present and checks if it holds a function type. Ensures
            /// getType, getNumFuncArguments, and getNumFuncResults can be called
            /// safely.
            LogicalResult verifyType() {
                auto type = getTypeAttr().getValue();

                if (!type.isa<FunctionType>()) {
                    return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
               }

                return success();
            }
    }];
}

def IDA_JacobianFunctionOp : IDA_Op<"jacobian_function",
    [IsolatedFromAbove, FunctionLike, AutomaticAllocationScope, Symbol]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 6> args;
            args.push_back($_builder.getF64Type());
            args.push_back(VariablesListType::get($_builder.getContext()));
            args.push_back(VariablesListType::get($_builder.getContext()));
            args.push_back(IndicesListType::get($_builder.getContext()));
            args.push_back($_builder.getF64Type());
            args.push_back($_builder.getI64Type());

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            FunctionType getType() {
                return getTypeAttr().getValue().cast<FunctionType>();
            }

            //===------------------------------------------------------------------===//
            // FunctionOpInterface Methods
            //===------------------------------------------------------------------===//

            /// Returns the argument types of this function.
            ArrayRef<Type> getArgumentTypes() {
                return getType().getInputs();
            }

            /// Returns the result types of this function.
            ArrayRef<Type> getResultTypes() {
                return getType().getResults();
            }

            //===------------------------------------------------------------------===//
            // SymbolOpInterface Methods
            //===------------------------------------------------------------------===//

            bool isDeclaration() {
                return isExternal();
            }

        private:
            // This trait needs access to the hooks defined below.
            friend class OpTrait::FunctionLike<JacobianFunctionOp>;

            /// Returns the number of arguments. This is a hook for
            /// OpTrait::FunctionLike.
            unsigned getNumFuncArguments() {
                return getType().getInputs().size();
            }

            /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
            unsigned getNumFuncResults() {
                return getType().getResults().size();
            }

            /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
            /// attribute is present and checks if it holds a function type. Ensures
            /// getType, getNumFuncArguments, and getNumFuncResults can be called
            /// safely.
            LogicalResult verifyType() {
                auto type = getTypeAttr().getValue();

                if (!type.isa<FunctionType>()) {
                    return emitOpError("requires '" + getTypeAttrName() + "' attribute of function type");
               }

                return success();
            }
    }];
}

def IDA_ReturnOp : IDA_Op<"return",
    [NoSideEffect,
     ParentOneOf<["ResidualFunctionOp", "JacobianFunctionOp"]>,
     ReturnLike, Terminator]>
{
    let summary = "Return operation";

    let description = [{
        The operation represents a return operation within a function.
        The operation takes variable number of operands and produces no results.
        The operand number and types must match the signature of the function
        that contains the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, llvm::None);
        }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def IDA_PrintStatisticsOp : IDA_Op<"print_statistics",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "TODO";

    let description = [{
        TODO
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

#endif // MARCO_DIALECTS_IDA_IDA_TD
