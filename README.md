# Modelica Compiler

*Cheatham's amendment of Conway's Law: If a group of N persons implements a [COBOL] compiler, there will be N-1 passes. Someone in the group has to be the manager.*

## Installation:
### Requirements:
* Boost graph
sudo apt install libboost-all-dev
* LLVM-9.0.0 
https://apt.llvm.org/
### Optionals
* doxigen to invoke the doc generation target
* genhtml and lcov to get the coverage report.

### DEBUG DEPENDENCIES 
Due to performance reasons llvm behaviour can be a lot different based upon the cmake flags used to compile llvm itself. If you require a debug build of this software read the following.
We raccomend to build the debug version of llvm from sources. Use the monorepo https://github.com/llvm/llvm-project 
Include the clang subproject since currently we are using that for the last stage of compilation. clang tools is not required. 
Do remember to checkout the llvm9 branch rather than building master.

A good talk about buidling llvm is (notice that not all steps are required if your are using the monorepo):
https://www.youtube.com/watch?v=uZI_Qla4pNA

#### -DLLVM_ABI_BREAKING_CHECKS
https://llvm.org/docs/CMake.html . That flags changes the behaviour of llvm::Expected to make it more strict in debug mode and faster in release mode. If you write code in release mode it may work release mode it will look like it is working but in reality errors will be silent and it will break in debug mode. DO NOT WRITE NEW CODE WITHOUT THIS FLAG ENABLED.

### -DLLVM_ENABLE_ASSERTIONS
In llvm preconditions of fuctions are written as assertions. This is because requirements should never be violated in correct code, thus it would wastefull to check them in release mode. On the other hand we do want those checks in debug mode since they usaully provide  a much more informative error message when triggered. Thus, a great deal of functions in llvm begins with the invocatio of assert and a debug build can be up to 20 times slower than a release build.

### Multiple versions of llvm
Notice that keeping separated multiple version of multiples types of llvm can be painfull. A trick is to never install llvm and to always provide the correct version to cmake. 

### Cloning
We are using google-test as testing framework. Thus git clone is not enough, you must
```bash
git clone https://github.com/drblallo/modelica
cd modelica
git submodule update --recursive --init
```
It will clone this repo as well as all dependencies that we must built with modelica.

#### Why is not boost graph a submodule as well?
boost libraries have a special status within cmake. They are not build with cmake but cmake itself provide a way of searching them in the system when they are used as dependecies. They are not as easy as it is for google-test to make them work.

#### Why is not boost llvm a submodule as well?
It would indeed must simplify the compatibility, it is not just for performace reasons.

### Building the release version
```bash
mkdir release 
cd release
cmake ..
make all
```


### Building the debug version
Minimal command
```bash
mkdir build 
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make all
```
Raccomended command
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILE
R=g++ -G Ninja -DCMAKE_EXE_LINKER_FLAGS='-fuse-ld=gold' -DCMAKE_CXX_FLAGS="-Werror -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter --coverage
" -DLLVM_DIR=HEEEEEERE --build ../
ninja all
```
#### What does this command does?
* CMAKE_BUILD_TYPE: set the version to debug mode
* CMAKE_EXPORT_COMPILE_COMMANDS: generates the compile_commands.json that contains informations about how compile each cpp file in the project. This file is often required by many tools and editors. In particular is needed by clang-tidy and clang-format. Read later about those tools.
* -G Ninja: select Ninja over make as a build tool. Ninja is faster
* CMAKE_EXE_LINKER_FLAGS="-fuse-ld=gold" select a different linker over ld that is much more performant. lld could be used as well if it is installed with llvm.
* DCMAKE_CXX_FLAGS: Enables to count all warning as werrs excepts for unused variables. we allow for unused variables because clang and gcc have different opinions about what is a unused variable. Furthermore this enables the coverage and allows to generate the coverage report.
* CMAKE_INSTALL_PREFIX selects were the library will be installed, read the installation sections to see more
* LLVM_DIR= the folder in which the finder of llvm will be searched. As an example if llvm is installed in usr/local/lib the correct value is "/usr/local/lib/cmake/llvm" 

### Installation
Once it has been built it can be installed. If you run make install or ninja install depending on your configurations modelica will be installed. When running
`cmake ..` you can configure the install location with `-DCMAKE_INSTALL_PREFIX=PATH`. After the installation the header, the libraries and the tools will be deployed in the target folder. Notice that debug libraries have a different name thus they can cohexists  with the release library in the install directory.

### Packaging
If the debug version is located in the prjectRoot/build folder and the release version is located in the projectRoot/release folder and the installation path is ./install for both and they have been build and installed you can create a package. Full command
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=./install
make all
cd ../
mkdir release
cd release
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make all
cd ..
cpack -G ZIP
``` 
It will produce a zip with debug and release libraries headers and tools

## Project Structure
The project is divided into tools and libs. Conceptually a lib does something and a tool lets you do the same things from command line.
### Libraries Structure
No nested lib are allowed, each lib must be placed in the ./lib folder. The reason for this choise is because dependencies are expressed with the cmake target_link_libraries command rather than with the position in the tree folder. Furthemore this allows for tools to autodetect which targets are avialable by just querying the subfolder names in lib. Should the need to visually check the library dependecies ever arise then use a cmake dependecy generator.

A library must be composed of
* A cmakelists.txt that invokes modelicaAddLibrary located in `./lib/libName/CMakeLists.txt`
* a include folder. This folder must be called as the library name and must be locaed in `./lib/libName/include/modelica/libName/`. All headers file in this folder will populate the public interface of this library. The repetition of the libName is necessary so that user of a header can include it as `#include "modelica/libName/HeaderName.hpp"` . We did not placed the header folder in a standalone include directory common to all libraries because in this way we can prevent libraries that do not depend on a second library to include its files.
* a src folder. Thid folder can be structured in any way, since it is private. Private headers can be placed here.
* a test folder. Reader in the tests sections for further informations

#### The modelicaAddLibraryMacro
This macro behaves like add_library, except it does the following as well
* Specifies that the include directory is pathToLib/include
* creates an alias for the lib called modelica::libName
* set the language level to c++17
* select which things will be installed where
* includes the test subfolder 

## Executables
As is the case for libraries all executables must be placed in the top level of the tool folder. A tool is composed of the following:
* folder located in ./tool/toolName that contains a CMakeLists.txt that invokes modelicaAddTool
* a src folder that must contain a Main.cpp file
* a test folder that can contain anything

### The modelicaAddTool Macro
The modelica add tool macro accepts a name for the tool as first argument and the name of libraries the tool will be depending on. The macro will
* Set the dependencies
* Create an alias called modelica::toolName for the tool
* set the language level to c++17
* select what will be installed where
* include the test subdirectory

## Testing

Each library and tool must include a test subdirectory. This subdirectory can use the modelicaAddTest macro. If it does then the folder must
* add src subdirectory that will contain the cpp file populated with google tests
* add a include directory that will be always private.

### The addModelicaTest macro
This macro will accepts the name of the executable that will contain the tests, it must be the name of the library itself. the postfix test will be added.
all other arguments must be the path to the cpp file that will compose the body of the test.
The macro will
* create the executable
* set were it must be installed.
* set the language level
* set a dependecy to the library with the same name

After building the test can be found in buildfolder/lib/libName/test/libNameTest

### Running tests
All tests can be run with `make(ninja) test` after it has been built.

### Continuous integration
The github ci is currently broken because ubuntu 18.04 they are using is shipping with a llvm9 that is different from the 9.0.0 and cannot be purged away because their own apt is broken due to missing mesa packages. Once we move to llvm10 it will work again.

## An example of what can be done
Once everything has been built and installed we can try the tools.
Consider the following simple example:
```modelica
model SimpleDer 
  final parameter Real tau = 5.0;
  Real[10] x(start = 0.0);
equation
  tau*der(x[1]) = 1.0;
  for i in 2:10 loop
	tau*der(x[i]) = 2.0*i;
  end for;
end SimpleDer;
```
Each der_x will be twice as much as the previous, thus we expect to see that each x will increase at twice the speed as the previous.

ASTDumper allows as to see how the ast looks like for this model:
```
ASTDumper model.mo
```
It will dump it before the type checking,, so many variables are of unkown type
To see the ast after the typechecking phase we can do

```
omcc -dt model.mo
```

To see what happens after the costant folding phase we can do
```
omcc -df model.mo
```
To see what the intermediate rappresentation looks like 
```
omcc -d model.mo
```
At this point we can save it to file with -o
```
omcc -d model.mo -o file.mod
```
If we are interested to see how it can be causalized we can use modmatch which will produce a dot file with the matched system. There flags to enable and disable some parts of the graph.
```
modmatch file.mod --dumpGraph=graph.dot
```
If we are just interested into the matched model we can do
```
omcc -dm model.mo
```
The left hand of the variables  describe which variable each equation has been matched too.

Now we can schedule the system, we can see the dependency graph with modsched
```
modsched --dumpGraph matched.mo > graph.dot 
```

If we are interested in the code after the matching we can do
```
 omcc -dsched model.mo 
 ```
Finally if we are interested into the llvm ir we can simply do
```
omcc -o out.bc model.mo
```
