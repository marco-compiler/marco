	.syntax unified
	.cpu cortex-m4
	.thumb

	.section .text



/* Reset vector. */
	.global Reset_Handler
	.type  Reset_Handler, %function
Reset_Handler:
	/* Copy .data from FLASH to RAM */
	ldr  r0, =_data
	ldr  r1, =_edata
	ldr  r2, =_etext
	cmp  r0, r1
	beq  nodata
dataloop:
	ldr  r3, [r2], #4
	str  r3, [r0], #4
	cmp  r1, r0
	bne  dataloop
nodata:
	/* Zero .bss */
	ldr  r0, =_bss_start
	ldr  r1, =_bss_end
	cmp  r0, r1
	beq  nobss
	movs r3, #0
bssloop:
	str  r3, [r0], #4
	cmp  r1, r0
	bne  bssloop
nobss:
	LDR.W R0, =0xE000ED88
	LDR R1, [R0]
	ORR R1, R1, #(0xF << 20)
	STR R1, [R0]
	DSB 
	ISB 
	/* Call global contructors for C++ */
	/* Can't use r0-r3 as the callee doesn't preserve them */
	ldr  r4, =__init_array_start
	ldr  r5, =__init_array_end
	cmp  r4, r5
	beq  noctor
ctorloop:
	ldr  r3, [r4], #4
	blx  r3
	cmp  r5, r4
	bne  ctorloop
noctor:
	/* Jump to main() */
	bl   main
	/*  If main() returns, endless loop */
loop:
	b    loop
	.size	Reset_Handler, .-Reset_Handler

/* Unimplemented interrupt function. */
	.global UnimplementedIrq
	.type  UnimplementedIrq, %function
UnimplementedIrq:
	b    UnimplementedIrq
	.size	UnimplementedIrq, .-UnimplementedIrq

/* Minimal interrupt vector table. Only the stack pointer and reset handler */
	.section .isr_vector
	.global __Vectors
__Vectors:
	.word     _stack_top
	.word     Reset_Handler
	.word     UnimplementedIrq /*NMI_Handler*/
	.word     UnimplementedIrq /*HardFault_Handler*/
	.word     UnimplementedIrq /*MemManage_Handler*/
	.word     UnimplementedIrq /*BusFault_Handler*/
	.word     UnimplementedIrq /*UsageFault_Handler*/
	.word     0
	.word     0
	.word     0
	.word     0
	.word     UnimplementedIrq /*SVC_Handler*/
	.word     UnimplementedIrq /*DebugMon_Handler*/
	.word     0
	.word     UnimplementedIrq /*PendSV_Handler*/
	.word     UnimplementedIrq /*SysTick_Handler*/ 
	/* External Interrupts */
	.word     UnimplementedIrq /*WWDG_IRQHandler*/                                     
	.word     UnimplementedIrq /*PVD_IRQHandler*/                 
	.word     UnimplementedIrq /*TAMP_STAMP_IRQHandler*/
	.word     UnimplementedIrq /*RTC_WKUP_IRQHandler*/
	.word     UnimplementedIrq /*FLASH_IRQHandler*/                                
	.word     UnimplementedIrq /*RCC_IRQHandler*/
	.word     UnimplementedIrq /*EXTI0_IRQHandler*/
	.word     UnimplementedIrq /*EXTI1_IRQHandler*/
	.word     UnimplementedIrq /*EXTI2_IRQHandler*/
	.word     UnimplementedIrq /*EXTI3_IRQHandler*/
	.word     UnimplementedIrq /*EXTI4_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream0_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream1_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream2_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream3_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream4_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream5_IRQHandler*/
	.word     UnimplementedIrq /*DMA1_Stream6_IRQHandler*/
	.word     UnimplementedIrq /*ADC_IRQHandler*/
	.word     UnimplementedIrq /*CAN1_TX_IRQHandler*/
	.word     UnimplementedIrq /*CAN1_RX0_IRQHandler*/
	.word     UnimplementedIrq /*CAN1_RX1_IRQHandler*/
	.word     UnimplementedIrq /*CAN1_SCE_IRQHandler*/
	.word     UnimplementedIrq /*EXTI9_5_IRQHandler*/
	.word     UnimplementedIrq /*TIM1_BRK_TIM9_IRQHandler*/
	.word     UnimplementedIrq /*TIM1_UP_TIM10_IRQHandler*/
	.word     UnimplementedIrq /*TIM1_TRG_COM_TIM11_IRQHandler*/
	.word     UnimplementedIrq /*TIM1_CC_IRQHandler*/
	.word     UnimplementedIrq /*TIM2_IRQHandler*/
	.word     UnimplementedIrq /*TIM3_IRQHandler*/
	.word     UnimplementedIrq /*TIM4_IRQHandler*/
	.word     UnimplementedIrq /*I2C1_EV_IRQHandler*/
	.word     UnimplementedIrq /*I2C1_ER_IRQHandler*/
	.word     UnimplementedIrq /*I2C2_EV_IRQHandler*/
	.word     UnimplementedIrq /*I2C2_ER_IRQHandler*/
	.word     UnimplementedIrq /*SPI1_IRQHandler*/
	.word     UnimplementedIrq /*SPI2_IRQHandler*/
	.word     UnimplementedIrq /*USART1_IRQHandler*/
	.word     _Z17USART2_IRQHandlerv
	/* There are many other interrupts beyond this, but we stop at USART2, which is te one we want */
