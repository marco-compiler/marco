CC  := arm-miosix-eabi-gcc
CXX := arm-miosix-eabi-g++
AS  := arm-miosix-eabi-as
CP  := arm-miosix-eabi-objcopy
CPU := -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16

AFLAGS   := $(CPU) 
CFLAGS   := $(CPU) -O0 -g -c -I./include/marco/driver/CMSIS/Include -I./include/marco/driver/CMSIS/Device/ST/STM32F4xx/Include -I./include/marco/driver
CXXFLAGS := $(CPU) -O0 -mbe32 -ggdb -fno-exceptions -fno-rtti -g -c -I./include/marco/driver/CMSIS/Include -I./include/marco/driver/CMSIS/Device/ST/STM32F4xx/Include -I./include/marco/driver -I../../include/public -I./include/marco/lib
LFLAGS   := $(CPU) -Wl,-T./src/linker.ld,-Map,./test/main.map #-nostdlib

SRC := ./src/startup.s ./test/math_test.cpp ./src/lib/Print.cpp ./src/serial.cpp ./src/pll_driver.cpp ./src/heap.cpp ./src/lib/StdMath.cpp ./src/lib/StdAssert.cpp

SRCS :=  ./src/startup.s ./src/main.cpp ./src/serial.cpp ./src/pll_driver.cpp ./src/heap.cpp ./src/stub.cpp ./src/lib/Print.cpp
MLIB := ./src/lib/BuiltInFunctions.cpp ./src/lib/Math.cpp ./src/lib/MemoryManagement.cpp ./src/lib/Profiling.cpp ./src/lib/Runtime.cpp ./src/lib/UtilityFunctions.cpp

OBJ := $(addsuffix .o, $(basename $(SRC))) 

OBJLIB := ./staticlib/Print.o ./staticlib/heap.o ./staticlib/StdMath.o ./staticlib/StdAssert.o ./staticlib/BuiltInFunctions.o ./staticlib/Math.o ./staticlib/MemoryManagement.o ./staticlib/Profiling.o ./staticlib/Runtime.o ./staticlib/UtilityFunctions.o 

OBJS := $(addsuffix .o, $(basename $(SRCS))) 

all: $(OBJ)
	make lib
	$(CXX) $(LFLAGS) -o math_test.elf $(OBJ) 
	$(CP) -O binary math_test.elf math_test.bin

sim: $(OBJS)
	@marco --omc-bypass --emit-llvm-ir -no-generate-main ./test/example.mo > ./test/example.ll
	@python3 64to32.py
	@llvm-as test/nomain.ll -o test/nomain.bc
	@clang++ -m32 --target=arm-none-eabi -march=armv7e-m -mcpu=cortex-m4 -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -fshort-enums test/nomain.bc -c -o test/nomain.o
	@make lib
	$(CXX) $(LFLAGS) -o main.elf $(OBJS) ./test/nomain.o -L. -lMARCORuntime 
	$(CP) -O binary main.elf main.bin


lib: $(MLIB) 
	mkdir staticlib
	$(CXX) $(CXXFLAGS) ./src/heap.cpp -o ./staticlib/heap.o
	$(CXX) $(CXXFLAGS) ./src/lib/Print.cpp -o ./staticlib/Print.o
	$(CXX) $(CXXFLAGS) ./src/lib/StdAssert.cpp -o ./staticlib/StdAssert.o
	$(CXX) $(CXXFLAGS) ./src/lib/StdMath.cpp -o ./staticlib/StdMath.o
	$(CXX) $(CXXFLAGS) ./src/lib/BuiltInFunctions.cpp -o ./staticlib/BuiltInFunctions.o
	$(CXX) $(CXXFLAGS) ./src/lib/Math.cpp -o ./staticlib/Math.o
	$(CXX) $(CXXFLAGS) ./src/lib/MemoryManagement.cpp -o ./staticlib/MemoryManagement.o
	$(CXX) $(CXXFLAGS) ./src/lib/Profiling.cpp -o ./staticlib/Profiling.o
	$(CXX) $(CXXFLAGS) ./src/lib/Runtime.cpp -nostdlib -o ./staticlib/Runtime.o
	$(CXX) $(CXXFLAGS) ./src/lib/UtilityFunctions.cpp -o ./staticlib/UtilityFunctions.o

	arm-miosix-eabi-ar -rcs libMARCORuntime.a $(OBJLIB)
clean:
	make cleanlib
	-rm $(OBJ) main.elf main.bin main.map

cleansim:

	rm test/example.ll test/nomain.ll test/nomain.bc  test/nomain.o 
	make cleanlib
	rm $(addsuffix .o, $(basename $(SRCS))) main.bin main.elf main.map

cleanlib:
	rm -r staticlib libMARCORuntime.a
	

%.o : %.s
	$(AS) $(AFLAGS) $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ 
