#include "pll_driver.h"
#include "registers.h"

#define MAX_FREQ 80 //84MHz is the max frequence for stm32f401re
#define DEF_FREQ  16 //16MHz if the default frequence for stm32f401re

#define PLLM 16

PLL_Driver::PLL_Driver(){
    this->setMaxFrequency();
};
/**
 * @brief set system clock frequency
 * 
 * @param frequency expressed in MHz
 * @param pllp 
 */
void PLL_Driver::setFrequency(unsigned int frequency, uint8_t pllp){

    uint8_t PLLP = 0b01; //default 4
    
    switch (pllp) 
    {
    case 4:
        break;
    case 2:
        PLLP = 0;
        break;
    case 6:
        PLLP = 0b10;
        break;
    case 8:
        PLLP = 0b11;
        break;
    default:
        break;
    }

    uint16_t PLLN = frequency >= MAX_FREQ ? 320 : frequency/PLLM/pllp;

    RCC -> CR |= RCC_CR_HSION;  
    while((RCC -> CR & RCC_CR_HSIRDY) != RCC_CR_HSIRDY){}
    RCC->CR &= ~(RCC_CR_PLLON);
    while((RCC->CR & RCC_CR_PLLRDY) == RCC_CR_PLLRDY){}
    RCC -> APB1ENR = 1 << 28;
    PWR -> CR = PWR_CR_VOS;
    FLASH -> ACR |= FLASH_ACR_ICEN | FLASH_ACR_PRFTEN | FLASH_ACR_DCEN  |FLASH_ACR_LATENCY_3WS;
    RCC -> PLLCFGR &= ~((0b11 << 16) | (0b111111111 << 6) | (0b111111));
    RCC -> PLLCFGR &= RCC_PLLCFGR_RST_VALUE;
    RCC->PLLCFGR |= (PLLP << 16) | (PLLN << 6) | (PLLM<< 0);
    RCC -> PLLCFGR |= RCC_PLLCFGR_PLLSRC_HSI;
    RCC -> CFGR = 0x00000000;
    RCC->CFGR |= RCC_CFGR_HPRE_DIV1;
    RCC->CFGR |= RCC_CFGR_PPRE1_DIV2;  //APB1 presacaler - Max freq is 40Mhz.
    RCC->CFGR |= RCC_CFGR_PPRE2_DIV1;  
    RCC -> CR |= RCC_CR_PLLON;
    while((RCC->CR & RCC_CR_PLLRDY) != RCC_CR_PLLRDY){}
    RCC -> CFGR &= ~(0b11 << 0);
    RCC -> CFGR |= RCC_CFGR_SW_PLL;
    while((RCC -> CFGR & RCC_CFGR_SWS_PLL) != RCC_CFGR_SWS_PLL ){}   

};

void PLL_Driver::setMaxFrequency(){
    setFrequency(MAX_FREQ, 4);
}


