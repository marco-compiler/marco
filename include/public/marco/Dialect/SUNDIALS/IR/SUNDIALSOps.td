#ifndef MARCO_DIALECT_SUNDIALS_IR_SUNDIALSOPS_TD
#define MARCO_DIALECT_SUNDIALS_IR_SUNDIALSOPS_TD

include "marco/Dialect/SUNDIALS/IR/SUNDIALSAttributes.td"
include "marco/Dialect/SUNDIALS/IR/SUNDIALSTypes.td"
include "marco/Dialect/SUNDIALS/IR/SUNDIALS.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===---------------------------------------------------------------------===//
// SUNDIALS operation definitions
//===---------------------------------------------------------------------===//

class SUNDIALS_Op<string mnemonic, list<Trait> traits = []> :
    Op<SUNDIALS_Dialect, mnemonic, traits>;

def SUNDIALS_VariableGetterOp : SUNDIALS_Op<"variable_getter",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface]>
{
    let summary = "Getter interface for an external variable.";

    let description = [{
        The `sundials.variable_getter` is a function-like operation that acts
        as a bridge between SUNDIALS and the custom dialect types.
        It allows to define how an array variable can be accessed and one of
        its scalar values retrieved.

        The function signature consists in a variadic list of arguments and one
        result.
        The first argument is the array variable (with its original type); the
        remaining ones consist in the indices of the scalar variable to be
        accessed.
        The return value is indeed the accessed scalar variable.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let results = (outs);

    let regions = (region
        AnyRegion:$body
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 2> args;
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayRef<BlockArgument> getVariableIndices();

        //===-------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===-------------------------------------------------------------===//

        bool isDeclaration()
        {
            return isExternal();
        }

        //===-------------------------------------------------------------===//
        // CallableOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        //===-------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }
    }];
}

def sundials_VariableSetterOp : SUNDIALS_Op<"variable_setter",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface]>
{
    let summary = "Setter interface for an external variable.";

    let description = [{
        The `sundials.variable_setter` is a function-like operation that acts
        as a bridge between sundials and the custom dialect types.
        It allows to define how an array variable can be accessed and one of
        its scalar values updated.

        The function signature consists in a variadic list of arguments.
        The first argument is the array variable (with its original type); the
        second argument is the value to be set; the remaining ones consist in
        the indices of the scalar variable to be accessed.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let results = (outs);

    let regions = (region
        AnyRegion:$body
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;
            args.push_back($_builder.getF64Type());
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, std::nullopt);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getValue();
        ArrayRef<BlockArgument> getVariableIndices();

        //===-------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===-------------------------------------------------------------===//

        bool isDeclaration()
        {
            return isExternal();
        }

        //===-------------------------------------------------------------===//
        // CallableOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        //===-------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }
    }];
}

def SUNDIALS_AccessFunctionOp : SUNDIALS_Op<"access_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface]>
{
    let summary = "Access function.";

    let description = [{
        The operation represents an access function for a variable, that is the
        way in which the indices of the variable are computed starting from the
        indices of the equation.

        Example:

        ```mlir
        sundials.access_function @foo(%arg0: index, %arg1: index) -> (index, index) {
            %0 = ... : index
            %1 = ... : index
            sundials.return %0, %1 : index, index
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let results = (outs);

    let regions = (region
        AnyRegion:$body
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "uint64_t":$equationRank, "uint64_t":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            SmallVector<Type, 2> results;

            for (size_t i = 0; i < variableRank; ++i) {
                results.push_back($_builder.getIndexType());
            }


            auto functionType = $_builder.getFunctionType(args, results);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayRef<BlockArgument> getEquationIndices();

        //===-------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===-------------------------------------------------------------===//

        bool isDeclaration()
        {
            return isExternal();
        }

        //===-------------------------------------------------------------===//
        // CallableOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        //===-------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }
    }];
}

def SUNDIALS_ReturnOp : SUNDIALS_Op<"return",
    [NoMemoryEffect,
     ParentOneOf<["VariableGetterOp", "VariableSetterOp", "AccessFunctionOp"]>,
     ReturnLike, Terminator]>
{
    let summary = "Return operation.";

    let description = [{
        The operation represents a return operation within a function.
        The operation takes variable number of operands and produces no
        results. The operand number and types must match the signature of the
        function that contains the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, std::nullopt);
        }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // MARCO_DIALECT_SUNDIALS_IR_SUNDIALSOPS_TD
