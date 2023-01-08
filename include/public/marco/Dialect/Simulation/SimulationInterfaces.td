#ifndef MARCO_DIALECTS_SIMULATION_SIMULATIONINTERFACES_TD
#define MARCO_DIALECTS_SIMULATION_SIMULATIONINTERFACES_TD

include "marco/Dialect/Simulation/SimulationDialect.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Simulation interface definitions
//===----------------------------------------------------------------------===//

class Simulation_Interface<string name> : OpInterface<name>
{
    let cppNamespace = "::mlir::simulation";
}

def Simulation_FunctionOpInterface
    : Simulation_Interface<"FunctionOpInterface">
{
    let methods = [
        InterfaceMethod<[{
            Returns the function argument types.
        }],
        "::llvm::ArrayRef<::mlir::Type>", "getArgumentTypes">,
        InterfaceMethod<[{
            Returns the function result types.
        }],
        "::llvm::ArrayRef<::mlir::Type>", "getResultTypes">,
    ];

    let extraSharedClassDeclaration = [{
        //===-------------------------------------------------------------===//
        // Body handling
        //===-------------------------------------------------------------===//

        /// Return the region containing the body of this function.
        ::mlir::Region& getFunctionBody()
        {
            return $_op->getRegion(0);
        }

        /// Return the list of blocks within the function body.
        ::mlir::Region::BlockListType& getBlocks()
        {
            return getFunctionBody().getBlocks();
        }

        /// Returns true if this function has no blocks within the body.
        bool empty()
        {
            return getFunctionBody().empty();
        }

        /// Push a new block to the back of the body region.
        void push_back(::mlir::Block* block)
        {
            getFunctionBody().push_back(block);
        }

        /// Push a new block to the front of the body region.
        void push_front(::mlir::Block* block)
        {
            getFunctionBody().push_front(block);
        }

        /// Return the last block in the body region.
        ::mlir::Block& back()
        {
            return getFunctionBody().back();
        }

        /// Return the first block in the body region.
        ::mlir::Block& front()
        {
            return getFunctionBody().front();
        }

        /// Add an entry block to an empty function, and set up the block
        /// arguments to match the signature of the function. The newly
        /// inserted entry block is returned.
        ::mlir::Block* addEntryBlock() {
            assert(empty() && "function already has an entry block");
            mlir::Block* entry = new Block();
            push_back(entry);

            llvm::ArrayRef<mlir::Type> inputTypes = $_op.getArgumentTypes();

            llvm::SmallVector<mlir::Location> locations(
                inputTypes.size(), $_op.getOperation()->getLoc());

            entry->addArguments(inputTypes, locations);
            return entry;
        }

        //===-------------------------------------------------------------===//
        // Type attribute handling
        //===-------------------------------------------------------------===//

        /// Returns the name of the attribute used for function types.
        static ::llvm::StringRef getTypeAttrName()
        {
            return ::mlir::simulation::function_interface_impl::getTypeAttrName();
        }

        /// Return the attribute containing the type of this function.
        ::mlir::TypeAttr getFunctionTypeAttr()
        {
            return this->getOperation()->template getAttrOfType<TypeAttr>(
                getTypeAttrName());
        }

        /// Return the type of this function.
        ::mlir::Type getFunctionType()
        {
            return getFunctionTypeAttr().getValue();
        }

        //===-------------------------------------------------------------===//
        // Argument and result handling
        //===-------------------------------------------------------------===//

        /// Returns the number of function arguments.
        unsigned int getNumArguments()
        {
            return $_op.getArgumentTypes().size();
        }

        /// Returns the number of function results.
        unsigned int getNumResults()
        {
            return $_op.getResultTypes().size();
        }

        /// Returns the entry block function argument at the given index.
        mlir::BlockArgument getArgument(unsigned int idx)
        {
            return getFunctionBody().getArgument(idx);
        }

        /// Support argument iteration.
        ::mlir::Region::BlockArgListType getArguments()
        {
            return getFunctionBody().getArguments();
        }
    }];
}

#endif // MARCO_DIALECTS_SIMULATION_SIMULATIONINTERFACES_TD
