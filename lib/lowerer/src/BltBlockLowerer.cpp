#include "modelica/lowerer/BltBlockLowerer.hpp"

#include "modelica/lowerer/IdaSolver.hpp"
#include "modelica/model/ModErrors.hpp"

using namespace std;
using namespace llvm;
using namespace modelica;

namespace modelica
{
	Expected<llvm::SmallVector<llvm::Value *, 3>> lowerBltBlock(
			LowererContext &info, const ModBltBlock &bltBlock)
	{
		assert(false && "To be implemented");					 // TODO
		return make_error<UnsolvableAlgebraicLoop>();	 // TODO
	}
}	 // namespace modelica
