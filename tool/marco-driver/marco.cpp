#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/InitLLVM.h"

extern int mc1_main(llvm::ArrayRef<const char *> argv);

int main(int argc, const char** argv)
{
  llvm::InitLLVM x(argc, argv);
  llvm::SmallVector<const char *, 256> args(argv, argv + argc);
  return mc1_main(llvm::makeArrayRef(args).slice(1));
}
